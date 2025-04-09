#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#define MAGIC_STRING "\x93NUMPY"
#define MAGIC_STRING_LEN 6
#define MAX_LINE_LENGTH 1024
#define NUM_ANTENNAE 196
#define WARPS_PER_BLOCK 7
#define FULL_MASK 0xffffffff

typedef struct
{
    float real;
    float imag;
} complex64;

typedef struct
{
    int index;
    float x_loc;
    float y_loc;
    float r;
} Antenna;

__global__ void beamform(complex64 *d_data, const float *__restrict__ weights, const float *__restrict__ phase_offset, int n_rows, int n_cols, complex64 *d_output)
{
    __shared__ complex64 shared_sum[WARPS_PER_BLOCK];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Each data point will have NUM_ANTENNAE threads associated with it
    // So we can figure out which time step and antenna we are associated with.
    int antenna = idx % NUM_ANTENNAE; // should be the same as the thread index in the block at this stage.

    int warp_num = threadIdx.x / 32;
    int thread_in_warp = threadIdx.x % 32;
    complex64 final_shared_sum;
    final_shared_sum.real = 0;
    final_shared_sum.imag = 0;

    complex64 sum;
    sum.real = 0;
    sum.imag = 0;

    if (idx < n_cols * n_rows)
    {
        // printf("Antennae %i: weight %f phase_offset %f\n", idx, weights[idx], phase_offset[idx]);

        sum.real += weights[antenna] * phase_offset[antenna] * d_data[idx].real;
        sum.imag += weights[antenna] * phase_offset[antenna] * d_data[idx].imag;
    }

    for (int offset = 16; offset > 0; offset /= 2)
    {
        sum.real += __shfl_down_sync(FULL_MASK, sum.real, offset);
        sum.imag += __shfl_down_sync(FULL_MASK, sum.imag, offset);
    }

    if (thread_in_warp == 0)
    {
        shared_sum[warp_num].real = sum.real;
        shared_sum[warp_num].imag = sum.imag;
    }

    __syncthreads();

    if (threadIdx.x < WARPS_PER_BLOCK)
    {
        final_shared_sum.real = shared_sum[threadIdx.x].real;
        final_shared_sum.imag = shared_sum[threadIdx.x].imag;
        for (int offset = 16; offset > 0; offset /= 2)
        {
            final_shared_sum.real += __shfl_down_sync(FULL_MASK, final_shared_sum.real, offset);
            final_shared_sum.imag += __shfl_down_sync(FULL_MASK, final_shared_sum.imag, offset);
        }
    }

    if (threadIdx.x == 0)
    {
        d_output[blockIdx.x].real = final_shared_sum.real;
        d_output[blockIdx.x].imag = final_shared_sum.imag;
    }
}

int extract_shape(const char *header, int *n_rows, int *n_cols)
{
    const char *shape_start = strstr(header, "'shape': (");
    if (!shape_start)
    {
        printf("Error: 'shape' not found in header.\n");
        return 0;
    }

    shape_start += strlen("'shape': (");

    int result = sscanf(shape_start, "%d, %d", n_rows, n_cols);
    if (result != 2)
    {
        printf("Error: Failed to parse shape.\n");
        return 0;
    }

    return 1;
}

void read_npy_file(const char *filename, complex64 **data, int *n_rows, int *n_cols)
{
    FILE *file = fopen(filename, "rb");
    if (!file)
    {
        perror("Error opening file");
        return;
    }

    // Step 1: Check magic string
    unsigned char magic[MAGIC_STRING_LEN];
    fread(magic, 1, MAGIC_STRING_LEN, file);
    if (memcmp(magic, MAGIC_STRING, MAGIC_STRING_LEN) != 0)
    {
        printf("Not a valid .npy file.\n");
        fclose(file);
        return;
    }

    // Step 2: Read version numbers (2 bytes)
    unsigned char version[2];
    fread(version, 1, 2, file);
    printf("File version: %d.%d\n", version[0], version[1]);

    // Step 3: Read header length (2 bytes)
    unsigned char header_len_bytes[2];
    fread(header_len_bytes, 1, 2, file);
    short header_len = (header_len_bytes[0] | (header_len_bytes[1] << 8));

    printf("Header length: %d\n", header_len);

    // Step 4: Read header
    char *header = (char *)malloc(header_len + 1);
    if (!header)
    {
        perror("Memory allocation error");
        fclose(file);
        return;
    }
    fread(header, 1, header_len, file);
    header[header_len] = '\0'; // Null-terminate the string

    printf("Header: %s\n", header);

    if (!extract_shape(header, n_rows, n_cols))
    {
        printf("Error extracting shape.\n");
        free(header);
        fclose(file);
        return;
    }
    printf("Shape: %d x %d\n", *n_rows, *n_cols);

    // Step 5: Read data
    int data_size = (*n_rows) * (*n_cols);

    // Use pinned memory to improve performance.
    cudaError_t err = cudaMallocHost((void **)data, data_size * sizeof(complex64));
    if (err != cudaSuccess)
    {
        perror("Memory allocation error");
        cudaFreeHost(header);
        fclose(file);
        return;
    }

    fread(*data, sizeof(complex64), data_size, file);

    // Print first 5 elements of the first 5 antennae as a sanity check.
    printf("Data (first 5 complex numbers of each thread):\n");
    for (int i = 0; i < 5 && i < *n_rows; i++)
    {
        printf("Time Step %i:\n", i);
        for (int j = 0; j < 5 && j < *n_cols; j++)
        {
            printf("Complex %d: %.2f + %.2fi\n", j, (*data)[i * (*n_cols) + j].real, (*data)[i * (*n_cols) + j].imag);
        }
    }

    free(header);
    fclose(file);
}

Antenna *read_antenna_map()
{
    FILE *file = fopen("../antenna_locations_only_used.csv", "r");
    if (!file)
    {
        perror("Could not open file");
        return NULL;
    }

    Antenna *antennae = NULL;
    cudaError_t err = cudaMallocHost((void **)&antennae, NUM_ANTENNAE * sizeof(Antenna));

    char line[MAX_LINE_LENGTH];
    // skip header
    fgets(line, sizeof(line), file);
    int count = 0;
    // Read line by line
    while (fgets(line, sizeof(line), file))
    {
        char *token = strtok(line, ",");

        antennae[count].index = count;
        antennae[count].x_loc = atof(token);

        token = strtok(NULL, ",");
        antennae[count].y_loc = atof(token);
        token = strtok(NULL, ",");
        antennae[count].r = atof(token);
        count++;
    }

    fclose(file);
    return antennae;
}

int main()
{
    Antenna *antennas = read_antenna_map();

    float phase_offset[NUM_ANTENNAE];
    float weights[NUM_ANTENNAE];
    for (int i = 0; i < NUM_ANTENNAE; i++)
    {
        // weights[i] = 1 / antennas[i].r;
        weights[i] = 1; // Make things easy to start with.
        phase_offset[i] = 1;
    }

    float *d_weights = NULL;
    cudaError_t err = cudaMalloc((void **)&d_weights, NUM_ANTENNAE * sizeof(float));
    if (err != cudaSuccess)
    {
        printf("CUDA memory allocation failed\n");
        return -1;
    }


    float *d_phase_offset = NULL;
    err = cudaMalloc((void **)&d_phase_offset, NUM_ANTENNAE * sizeof(float));
    if (err != cudaSuccess)
    {
        printf("CUDA memory allocation failed\n");
        cudaFree(d_weights);
        cudaFreeHost(antennas);
        return -1;
    }

    const char *filename = "../antenna_data_transposed.npy";
    complex64 *data = NULL;
    int n_rows, n_cols;
    read_npy_file(filename, &data, &n_rows, &n_cols);
    printf("data %f, %f\n", data[0].real, data[0].imag);
    printf("data has shape %i x %i\n", n_rows, n_cols);

    complex64 *d_data = NULL;
    err = cudaMalloc((void **)&d_data, n_rows * n_cols * sizeof(complex64));
    if (err != cudaSuccess)
    {
        printf("CUDA memory allocation failed\n");
        cudaFreeHost(data);
        cudaFreeHost(antennas);
        cudaFree(d_weights);
        cudaFree(d_phase_offset);
        return -1;
    }

    complex64 *d_output = NULL;
    err = cudaMalloc((void **)&d_output, n_rows * sizeof(complex64));
    if (err != cudaSuccess)
    {
        printf("CUDA memory allocation failed\n");
        cudaFreeHost(data);
        cudaFreeHost(antennas);
        cudaFree(d_data);
        cudaFree(d_weights);
        cudaFree(d_phase_offset);
        return -1;
    }

    complex64 *output = NULL;
    err = cudaMallocHost((void **)&output, n_rows * sizeof(complex64));
    if (err != cudaSuccess)
    {
        printf("CUDA memory allocation failed\n");
        cudaFreeHost(data);
        cudaFreeHost(antennas);
        cudaFree(d_data);
        cudaFree(d_output);
        cudaFree(d_phase_offset);
        cudaFree(d_weights);
        return -1;
    }

    cudaStream_t stream;
    err = cudaStreamCreate(&stream);
    if (err != cudaSuccess)
    {
        printf("CUDA stream creation failed\n");
        cudaFree(d_data);
        cudaFree(d_output);
        cudaFree(d_weights);
        cudaFree(d_phase_offset);
        cudaFreeHost(data);
        cudaFreeHost(output);
        cudaFreeHost(antennas);
        return -1;
    }

    err = cudaMemcpyAsync(d_weights, weights, NUM_ANTENNAE * sizeof(float), cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess)
    {
        printf("CUDA memory copy failed\n");
        cudaFree(d_weights);
        cudaFree(d_data);
        cudaFree(d_phase_offset);
        cudaFree(d_output);
        cudaFreeHost(data);
        cudaFreeHost(output);
        cudaFreeHost(antennas);
        cudaStreamDestroy(stream);
        return -1;
    }

    err = cudaMemcpyAsync(d_phase_offset, phase_offset, NUM_ANTENNAE * sizeof(float), cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess)
    {
        printf("CUDA memory copy failed\n");
        cudaFree(d_weights);
        cudaFree(d_data);
        cudaFree(d_phase_offset);
        cudaFree(d_output);
        cudaFreeHost(data);
        cudaFreeHost(output);
        cudaFreeHost(antennas);
        cudaStreamDestroy(stream);
        return -1;
    }

    err = cudaMemcpyAsync(d_data, data, n_rows * n_cols * sizeof(complex64), cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess)
    {
        printf("CUDA memory copy failed\n");
        cudaFree(d_data);
        cudaFree(d_weights);
        cudaFree(d_phase_offset);
        cudaFree(d_output);
        cudaFreeHost(data);
        cudaFreeHost(output);
        cudaFreeHost(antennas);
        cudaStreamDestroy(stream);
        return -1;
    }
    // Check for kernel execution errors
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA kernel launch failed: %s\n", cudaGetErrorString(err));
    }

    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess)
    {
        printf("CUDA stream synchronization failed\n");
    }

    beamform<<<n_rows, NUM_ANTENNAE>>>(d_data, d_weights, d_phase_offset, n_rows, n_cols, d_output);

    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess)
    {
        printf("CUDA stream synchronization failed\n%s\n", cudaGetErrorString(err));
    }

    err = cudaMemcpyAsync(output, d_output, n_rows * sizeof(complex64), cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess)
    {
        printf("CUDA memory copy failed\n");
        cudaFree(d_weights);
        cudaFree(d_data);
        cudaFree(d_phase_offset);
        cudaFree(d_output);
        cudaFreeHost(data);
        cudaFreeHost(output);
        cudaStreamDestroy(stream);
        return -1;
    }

    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess)
    {
        printf("CUDA stream synchronization failed\n%s\n", cudaGetErrorString(err));
    }

    printf("First 5 values are...");
    for (int i = 0; i < 5; i++)
    {
        printf("%f + %fi\n", output[i].real, output[i].imag);
    }

    printf("Last value is...");
    printf("%f + %fi\n", output[n_rows - 1].real, output[n_rows - 1].imag);

    cudaFree(d_data);
    cudaFree(d_weights);
    cudaFree(d_phase_offset);
    cudaFree(d_output);
    cudaFreeHost(antennas);
    cudaFreeHost(data);
    cudaFreeHost(output);
    cudaStreamDestroy(stream);
    return 0;
}