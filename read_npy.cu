#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#define MAGIC_STRING "\x93NUMPY"
#define MAGIC_STRING_LEN 6

typedef struct
{
    float real;
    float imag;
} complex64;

__global__ void print_first_five_elements(complex64 *d_data, int n_rows, int n_cols)
{
    // This kernel acts as a sanity check and will output the first 5 vals of each
    // antenna datastream.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < 5)
    {
        printf("Time step %d:\n", idx);
        for (int j = 0; j < 5 && j < n_cols; j++)
        {
            int index = idx * n_cols + j;
            printf("Complex %d,%d: %.2f + %.2fi\n", idx, j, d_data[index].real, d_data[index].imag);
        }
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
        printf("Antenna %i:\n", i);
        for (int j = 0; j < 5 && j < *n_cols; j++)
        {
            printf("Complex %d: %.2f + %.2fi\n", j, (*data)[j * (*n_cols) + i].real, (*data)[j * (*n_cols) + i].imag);
        }
    }

    free(header);
    fclose(file);
}

int main()
{
    const char *filename = "../antenna_data_transposed.npy";
    complex64 *data = NULL;
    int n_rows, n_cols;
    read_npy_file(filename, &data, &n_rows, &n_cols);
    printf("data %f, %f\n", data[0].real, data[0].imag);
    printf("data has shape %i x %i\n", n_rows, n_cols);

    complex64 *d_data = NULL;
    cudaError_t err = cudaMalloc((void **)&d_data, n_rows * n_cols * sizeof(complex64));
    if (err != cudaSuccess)
    {
        printf("CUDA memory allocation failed\n");
        cudaFreeHost(data);
        return -1;
    }

    cudaStream_t stream;
    err = cudaStreamCreate(&stream);
    if (err != cudaSuccess)
    {
        printf("CUDA stream creation failed\n");
        cudaFree(d_data);
        cudaFreeHost(data);
        return -1;
    }

    err = cudaMemcpyAsync(d_data, data, n_rows * n_cols * sizeof(complex64), cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess)
    {
        printf("CUDA memory copy failed\n");
        cudaFree(d_data);
        cudaFreeHost(data);
        cudaStreamDestroy(stream);
        return -1;
    }
    // Define kernel execution configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (n_rows + threadsPerBlock - 1) / threadsPerBlock; // Round up the number of blocks

    printf("CUDA Version...\n");

    // Launch kernel to print first 5 elements of each antenna
    print_first_five_elements<<<blocksPerGrid, threadsPerBlock>>>(d_data, n_rows, n_cols);

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

    cudaFree(d_data);
    cudaFreeHost(data);
    cudaStreamDestroy(stream);
    return 0;
}