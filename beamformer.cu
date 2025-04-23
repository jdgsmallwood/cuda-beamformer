#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

#define MAGIC_STRING "\x93NUMPY"
#define MAGIC_STRING_LEN 6
#define MAX_LINE_LENGTH 1024
#define NUM_ANTENNAS 196
#define NUM_BEAMS 5
#define WARPS_PER_BLOCK 7
#define FULL_MASK 0xffffffff

typedef struct
{
    int index;
    float x_loc;
    float y_loc;
    float r;
} Antenna;

typedef struct
{
    float2 data[NUM_BEAMS * NUM_ANTENNAS];
} float2_beamarray;

typedef struct
{
    float2 data[NUM_BEAMS];
} float2_single_beamarray;

__global__ void beamform(const float2 *__restrict__ d_data, float2 *__restrict__ d_output, const float2_beamarray __restrict__ *d_weights_and_phase)
{
    __shared__ float2 partial_sum[NUM_BEAMS][WARPS_PER_BLOCK];
    float2 sum;
    const float2 data = d_data[blockIdx.x * blockDim.x + threadIdx.x];
    float2_single_beamarray weights_and_phase;

#pragma unroll
    for (int i = 0; i < NUM_BEAMS; i++)
    {
        // __ldg recommends that this be cached to read-only memory
        weights_and_phase.data[i] = __ldg(&(d_weights_and_phase->data[i * NUM_ANTENNAS + threadIdx.x]));
    }

#pragma unroll
    for (int beam = 0; beam < NUM_BEAMS; beam++)
    {
        sum.x = weights_and_phase.data[beam].y * data.x - data.y * weights_and_phase.data[beam].x;
        sum.y = data.x * weights_and_phase.data[beam].x + data.y * weights_and_phase.data[beam].y;

#pragma unroll
        for (int offset = 16; offset > 0; offset /= 2)
        {
            sum.x += __shfl_down_sync(FULL_MASK, sum.x, offset);
            sum.y += __shfl_down_sync(FULL_MASK, sum.y, offset);
        }

        // Is it the first thread in the warp? If so write to shared memory.
        if (threadIdx.x % 32 == 0)
        {
            partial_sum[beam][threadIdx.x / 32] = sum;
        }
    }

    __syncthreads();

    if (threadIdx.x < NUM_BEAMS)
    {
        sum = partial_sum[threadIdx.x][0];
#pragma unroll
        for (int i = 1; i < WARPS_PER_BLOCK; ++i)
        {
            sum.x += partial_sum[threadIdx.x][i].x;
            sum.y += partial_sum[threadIdx.x][i].y;
        }
        d_output[blockIdx.x * NUM_BEAMS + threadIdx.x] = sum;
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

void read_npy_file(const char *filename, float2 **data, int *n_rows, int *n_cols)
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
    cudaError_t err = cudaMallocHost((void **)data, data_size * sizeof(float2));
    if (err != cudaSuccess)
    {
        perror("Memory allocation error");
        cudaFreeHost(header);
        fclose(file);
        return;
    }

    fread(*data, sizeof(float2), data_size, file);

    // Print first 5 elements of the first 5 antennae as a sanity check.
    printf("Data (first 5 complex numbers of each thread):\n");
    for (int i = 0; i < 5 && i < *n_rows; i++)
    {
        printf("Time Step %i:\n", i);
        for (int j = 0; j < 5 && j < *n_cols; j++)
        {
            printf("Complex %d: %.2f + %.2fi\n", j, (*data)[i * (*n_cols) + j].x, (*data)[i * (*n_cols) + j].y);
        }
    }

    free(header);
    fclose(file);
}

Antenna *read_antenna_map()
{
    FILE *file = fopen("/fred/oz002/jsmallwo/antenna_locations_only_used.csv", "r");
    if (!file)
    {
        perror("Could not open file");
        return NULL;
    }

    Antenna *antennae = NULL;
    cudaError_t err = cudaMallocHost((void **)&antennae, NUM_ANTENNAS * sizeof(Antenna));

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

    float phase_offset[NUM_BEAMS * NUM_ANTENNAS];
    float weights[NUM_BEAMS * NUM_ANTENNAS];
    float2_beamarray weights_and_phase;

    float sin_phase, cos_phase;
    for (int beam = 0; beam < NUM_BEAMS; beam++)
    {
        for (int i = 0; i < NUM_ANTENNAS; i++)
        {
            // weights[i] = 1 / antennas[i].r;
            int offset = i * NUM_BEAMS + beam;
            int ant_offset = beam * NUM_ANTENNAS + i;
            weights[offset] = beam % 4; // Make things easy to start with.
            phase_offset[offset] = 0;

            sincosf(phase_offset[offset], &sin_phase, &cos_phase);
            weights_and_phase.data[ant_offset] = {sin_phase * weights[offset], cos_phase * weights[offset]};
        }
    }

    float2_beamarray *d_weights_and_phase = NULL;
    cudaError_t err = cudaMalloc((void **)&d_weights_and_phase, sizeof(float2_beamarray));
    if (err != cudaSuccess)
    {
        printf("CUDA copy to phase offset symbol failed.\n");
        return -1;
    }

    const char *filename = "/fred/oz002/jsmallwo/antenna_data_transposed.npy";
    float2 *data = NULL;
    int n_rows, n_cols;
    read_npy_file(filename, &data, &n_rows, &n_cols);
    printf("data %f, %f\n", data[0].x, data[0].y);
    printf("data has shape %i x %i\n", n_rows, n_cols);

    float2 *d_data = NULL;
    err = cudaMalloc((void **)&d_data, n_rows * n_cols * sizeof(float2));
    if (err != cudaSuccess)
    {
        printf("CUDA memory allocation failed\n");
        cudaFreeHost(data);
        cudaFreeHost(antennas);
        cudaFree(d_weights_and_phase);
        return -1;
    }

    float2 *d_output = NULL;
    err = cudaMalloc((void **)&d_output, n_rows * NUM_BEAMS * sizeof(float2));
    if (err != cudaSuccess)
    {
        printf("CUDA memory allocation failed\n");
        cudaFreeHost(data);
        cudaFreeHost(antennas);
        cudaFree(d_data);
        cudaFree(d_weights_and_phase);
        return -1;
    }

    float2 *output = NULL;
    err = cudaMallocHost((void **)&output, n_rows * NUM_BEAMS * sizeof(float2));
    if (err != cudaSuccess)
    {
        printf("CUDA memory allocation failed\n");
        cudaFreeHost(data);
        cudaFreeHost(antennas);
        cudaFree(d_data);
        cudaFree(d_output);
        cudaFree(d_weights_and_phase);
        return -1;
    }
    cudaStream_t stream;
    err = cudaStreamCreate(&stream);
    if (err != cudaSuccess)
    {
        printf("CUDA stream creation failed\n");
        cudaFree(d_data);
        cudaFree(d_output);
        cudaFree(d_weights_and_phase);
        cudaFreeHost(data);
        cudaFreeHost(output);
        cudaFreeHost(antennas);
        return -1;
    }

    err = cudaMemcpyAsync(d_weights_and_phase, &weights_and_phase, sizeof(float2_beamarray), cudaMemcpyHostToDevice, stream);

    if (err != cudaSuccess)
    {
        printf("CUDA memory copy failed\n");
        cudaFree(d_data);
        cudaFree(d_weights_and_phase);
        cudaFree(d_output);
        cudaFreeHost(data);
        cudaFreeHost(output);
        cudaFreeHost(antennas);
        cudaStreamDestroy(stream);
        return -1;
    }
    err = cudaMemcpyAsync(d_data, data, n_rows * n_cols * sizeof(float2), cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess)
    {
        printf("CUDA memory copy failed\n");
        cudaFree(d_data);
        cudaFree(d_weights_and_phase);
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

    beamform<<<dim3(n_rows, 1), NUM_ANTENNAS, 0, stream>>>(d_data, d_output, d_weights_and_phase);

    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess)
    {
        printf("CUDA stream synchronization failed\n%s\n", cudaGetErrorString(err));
    }

    err = cudaMemcpyAsync(output, d_output, n_rows * NUM_BEAMS * sizeof(float2), cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess)
    {
        printf("CUDA memory copy failed\n");
        cudaFree(d_weights_and_phase);
        cudaFree(d_data);
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

    for (int beam = 0; beam < NUM_BEAMS; beam++)
    {
        printf("First 5 values of beam %i are...\n", beam);
        for (int i = 0; i < 5; i++)
        {
            printf("%f + %fi\n", output[i * n_rows + beam].x, output[i * n_rows + beam].y);
        }

    }

    cudaFree(d_data);
    cudaFree(d_weights_and_phase);
    cudaFree(d_output);
    cudaFree(d_output);
    cudaFreeHost(antennas);
    cudaFreeHost(data);
    cudaFreeHost(output);
    cudaFreeHost(output);
    cudaStreamDestroy(stream);
    return 0;
}