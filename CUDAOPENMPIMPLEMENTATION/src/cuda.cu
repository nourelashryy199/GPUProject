#include "cuda.cuh"
#include <cuda_runtime.h>
#include <math.h>

#define CUDA_BLOCK_W 16
#define CUDA_BLOCK_H 16

#define CG_BLOCK_W CUDA_BLOCK_W
#define CG_BLOCK_H CUDA_BLOCK_H

#define CG_TILE_W (CG_BLOCK_W + 2)
#define CG_TILE_H (CG_BLOCK_H + 2)

#define CG_THREADS_PER_BLOCK (CG_BLOCK_W * CG_BLOCK_H)
#define CG_WARPS_PER_BLOCK (CG_THREADS_PER_BLOCK / 32)

#define EMBOSS_BLOCK_W CUDA_BLOCK_W
#define EMBOSS_BLOCK_H CUDA_BLOCK_H

#define EMBOSS_TILE_W (EMBOSS_BLOCK_W + 2)
#define EMBOSS_TILE_H (EMBOSS_BLOCK_H + 2)
#define EMBOSS_THREADS_PER_BLOCK (EMBOSS_BLOCK_W * EMBOSS_BLOCK_H)

__constant__ unsigned char d_glider1[9];
__constant__ unsigned char d_glider2[9];

//static void printDeviceProperties() {  //function to get the specs of device to make informed memory-related decisions
//    cudaDeviceProp prop;
//    cudaGetDeviceProperties(&prop, 0);
//
//    printf("GPU: %s\n", prop.name);
//    printf("Compute Capacity: %d.%d\n", prop.major, prop.minor);
//    printf("Max Threads Per Block: %d\n", prop.maxThreadsPerBlock);
//    printf("Max Threads Per Streaming Multiprocessor(SM): %d\n", prop.maxThreadsPerMultiProcessor);
//    printf("Warp size: %d\n", prop.warpSize);
//    printf("Shared memory per block: %zu bytes\n", prop.sharedMemPerBlock);
//    printf("Number of Streaming Multiprocessors (SMs): %d\n", prop.multiProcessorCount);
//    printf("Number of Registers per Block: %d\n", prop.regsPerBlock);
//}
//static bool printed = false;
//static void printDevicePropertiesOnce() {
//    if (!printed) {
//        printf("\n CUDA DEVICE PROPERTIES: \n");
//        printDeviceProperties();
//        printf("=============================");
//        printed = true;
//    }
//}

__device__ __forceinline__ bool areSame(
    const unsigned char tile[CG_TILE_H][CG_TILE_W],
    int local_x,
    int local_y,
    const unsigned char* pattern,
    int reflect,
    int rotate
) {
    for (int gy = 0; gy < 3; gy++) {
        for (int gx = 0; gx < 3; gx++) {

            int glider_x = reflect ? (2 - gx) : gx;
            int glider_y = gy;

            for (int r = 0; r < rotate; r++) {
                int temp_x = glider_x;
                glider_x = glider_y;
                glider_y = 2 - temp_x;
            }

            unsigned char cell =
                tile[local_y + gy][local_x + gx] ? 1 : 0;

            unsigned char expected =
                pattern[glider_y * 3 + glider_x];

            if (cell != expected) {
                return false;
            }
        }
    }

    return true;
}
__global__ void countGlidersKernel(
    const unsigned char* cells,
    size_t width,
    size_t height,
    unsigned long long* count
) {
    __shared__ unsigned char tile[CG_TILE_H][CG_TILE_W];
    __shared__ unsigned int warpCounts[CG_WARPS_PER_BLOCK];

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;

    const int local_x = threadIdx.x;
    const int local_y = threadIdx.y;

    const size_t block_start_x = blockIdx.x * CG_BLOCK_W;
    const size_t block_start_y = blockIdx.y * CG_BLOCK_H;

    for (int i = tid; i < CG_TILE_W * CG_TILE_H; i += CG_THREADS_PER_BLOCK) {
        int tile_y = i / CG_TILE_W;
        int tile_x = i % CG_TILE_W;

        size_t global_x = block_start_x + tile_x;
        size_t global_y = block_start_y + tile_y;

        if (global_x < width && global_y < height) {
            tile[tile_y][tile_x] =
                cells[global_y * width + global_x];
        }
        else {
            tile[tile_y][tile_x] = 0;
        }
    }

    __syncthreads();

    const size_t out_width = width - 2;
    const size_t out_height = height - 2;

    const size_t x = block_start_x + local_x;
    const size_t y = block_start_y + local_y;

    bool found = false;

    if (x < out_width && y < out_height) {

        for (int reflect = 0; reflect < 2 && !found; reflect++) {
            for (int rotate = 0; rotate < 4 && !found; rotate++) {
                found = areSame(
                    tile,
                    local_x,
                    local_y,
                    d_glider1,
                    reflect,
                    rotate
                );
            }
        }

        for (int reflect = 0; reflect < 2 && !found; reflect++) {
            for (int rotate = 0; rotate < 4 && !found; rotate++) {
                found = areSame(
                    tile,
                    local_x,
                    local_y,
                    d_glider2,
                    reflect,
                    rotate
                );
            }
        }
    }

    const int lane = tid % 32;
    const int warpId = tid / 32;

    unsigned int mask = __ballot_sync(0xffffffff, found);
    unsigned int warpTotal = __popc(mask);

    if (lane == 0) {
        warpCounts[warpId] = warpTotal;
    }

    __syncthreads();

    if (tid == 0) {
        unsigned int blockTotal = 0;

        for (int i = 0; i < CG_WARPS_PER_BLOCK; i++) {
            blockTotal += warpCounts[i];
        }

        atomicAdd(count, (unsigned long long)blockTotal);
    }
}

uint64_t cuda_countGliders(
    const unsigned char* cells,
    const size_t width,
    const size_t height
) {
    if (width < 3 || height < 3) {
        return 0;
    }

    static bool copied = false;

    if (!copied) {
        static const unsigned char h_glider1[9] = {
            1, 0, 1,
            1, 1, 0,
            0, 0, 0
        };

        static const unsigned char h_glider2[9] = {
            0, 1, 0,
            1, 0, 0,
            1, 0, 1
        };

        cudaMemcpyToSymbol(d_glider1, h_glider1, sizeof(h_glider1));
        cudaMemcpyToSymbol(d_glider2, h_glider2, sizeof(h_glider2));

        copied = true;
    }

    unsigned char* d_cells = nullptr;
    unsigned long long* d_count = nullptr;

    size_t imageSize = width * height * sizeof(unsigned char);

    cudaMalloc(&d_cells, imageSize);
    cudaMalloc(&d_count, sizeof(unsigned long long));

    cudaMemcpy(d_cells, cells, imageSize, cudaMemcpyHostToDevice);
    cudaMemset(d_count, 0, sizeof(unsigned long long));

    dim3 block(CG_BLOCK_W, CG_BLOCK_H);

    dim3 grid(
        (width - 2 + CG_BLOCK_W - 1) / CG_BLOCK_W,
        (height - 2 + CG_BLOCK_H - 1) / CG_BLOCK_H
    );

    countGlidersKernel << <grid, block >> > (
        d_cells,
        width,
        height,
        d_count
        );

    cudaDeviceSynchronize();

    unsigned long long h_count = 0;

    cudaMemcpy(
        &h_count,
        d_count,
        sizeof(unsigned long long),
        cudaMemcpyDeviceToHost
    );

    cudaFree(d_cells);
    cudaFree(d_count);

    return (uint64_t)h_count;
}
__global__ void histogram_kernel(
    const int* numbers,
    size_t length,
    int bin_width,
    int* output,
    size_t histogram_len
) {
    extern __shared__ int local_hist[];

    for (size_t j = threadIdx.x; j < histogram_len; j += blockDim.x) {
        local_hist[j] = 0;
    }

    __syncthreads();

    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < length) {
        int bin = numbers[i] / bin_width;

        if (bin >= 0 && (size_t)bin < histogram_len) {
            atomicAdd(&local_hist[bin], 1);
        }
    }

    __syncthreads();

    for (size_t j = threadIdx.x; j < histogram_len; j += blockDim.x) {
        atomicAdd(&output[j], local_hist[j]);
    }
}
size_t cuda_histogram(const int *numbers, size_t length, int bin_width, int *output) {
    //printDevicePropertiesOnce();
    const size_t HISTOGRAM_LEN =
        (size_t)ceil(HISTOGRAM_MAX_VALUE / (double)bin_width);

    int* d_numbers = nullptr;
    int* d_output = nullptr;

    cudaMalloc(&d_numbers, sizeof(int) * length);
    cudaMalloc(&d_output, sizeof(int) * HISTOGRAM_LEN);

    cudaMemcpy(d_numbers, numbers, sizeof(int) * length, cudaMemcpyHostToDevice);

    cudaMemset(d_output, 0, sizeof(int) * HISTOGRAM_LEN);

    const int THREADS_PER_BLOCK = 256;
    const int BLOCKS =
        (int)((length + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

    size_t shared_mem_size = sizeof(int) * HISTOGRAM_LEN;

    histogram_kernel << <BLOCKS, THREADS_PER_BLOCK, shared_mem_size >> > (
        d_numbers,
        length,
        bin_width,
        d_output,
        HISTOGRAM_LEN
        );

    cudaDeviceSynchronize();

    cudaMemcpy(output, d_output, sizeof(int) * HISTOGRAM_LEN, cudaMemcpyDeviceToHost);

    cudaFree(d_numbers);
    cudaFree(d_output);

    return HISTOGRAM_LEN;
}
__constant__ float d_emboss_kernel[9];

__device__ unsigned char clamp_to_uchar(float value) {
    value = value < 0.0f ? 0.0f : value;
    value = value > 255.0f ? 255.0f : value;
    return (unsigned char)value;
}

__global__ void emboss_kernel(
    const unsigned char* pixels,
    size_t width,
    size_t height,
    unsigned char* output
) {
    __shared__ unsigned char tile[EMBOSS_TILE_H][EMBOSS_TILE_W][3];

    const int out_width = (int)width - 2;
    const int out_height = (int)height - 2;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    const int block_start_x = blockIdx.x * EMBOSS_BLOCK_W;
    const int block_start_y = blockIdx.y * EMBOSS_BLOCK_H;

    for (int i = tid; i < EMBOSS_TILE_W * EMBOSS_TILE_H; i += EMBOSS_THREADS_PER_BLOCK) {
        const int tile_y = i / EMBOSS_TILE_W;
        const int tile_x = i % EMBOSS_TILE_W;

        const int global_x = block_start_x + tile_x;
        const int global_y = block_start_y + tile_y;

        if (global_x < width && global_y < height) {
            const int global_offset = (global_y * width + global_x) * 3;

            tile[tile_y][tile_x][0] = pixels[global_offset + 0];
            tile[tile_y][tile_x][1] = pixels[global_offset + 1];
            tile[tile_y][tile_x][2] = pixels[global_offset + 2];
        }
        else {
            tile[tile_y][tile_x][0] = 0;
            tile[tile_y][tile_x][1] = 0;
            tile[tile_y][tile_x][2] = 0;
        }
    }

    __syncthreads();

    const int x = block_start_x + tx;
    const int y = block_start_y + ty;

    if (x >= out_width || y >= out_height) {
        return;
    }

    float pixel_sum = 0.0f;

    for (int ky = 0; ky < 3; ++ky) {
        for (int kx = 0; kx < 3; ++kx) {
            const unsigned char R = tile[ty + ky][tx + kx][0];
            const unsigned char G = tile[ty + ky][tx + kx][1];
            const unsigned char B = tile[ty + ky][tx + kx][2];

            const float grey =
                (0.2126f * R) + (0.7152f * G) + (0.0722f * B);

            pixel_sum += grey * d_emboss_kernel[ky * 3 + kx];
        }
    }

    pixel_sum += 128.0f;

    output[y * out_width + x] = clamp_to_uchar(pixel_sum);
}

void cuda_emboss(
    const unsigned char* pixels,
    const size_t width,
    const size_t height,
    unsigned char* output
) {
    //printDevicePropertiesOnce();

    const size_t input_size = width * height * 3 * sizeof(unsigned char);
    const size_t out_width = width - 2;
    const size_t out_height = height - 2;
    const size_t output_size = out_width * out_height * sizeof(unsigned char);

    unsigned char* d_pixels = nullptr;
    unsigned char* d_output = nullptr;

    static const float h_emboss_kernel[9] = {
        -2, -1,  0,
        -1,  0,  1,
         0,  1,  2
    };

    cudaMalloc(&d_pixels, input_size);
    cudaMalloc(&d_output, output_size);

    cudaMemcpy(d_pixels, pixels, input_size, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_emboss_kernel, h_emboss_kernel, sizeof(float) * 9);

    dim3 threadsPerBlock(EMBOSS_BLOCK_W, EMBOSS_BLOCK_H);

    dim3 blocksPerGrid(
        (out_width + EMBOSS_BLOCK_W - 1) / EMBOSS_BLOCK_W,
        (out_height + EMBOSS_BLOCK_H - 1) / EMBOSS_BLOCK_H
    );

    emboss_kernel << <blocksPerGrid, threadsPerBlock >> > (
        d_pixels,
        width,
        height,
        d_output
        );

    cudaDeviceSynchronize();

    cudaMemcpy(output, d_output, output_size, cudaMemcpyDeviceToHost);

    cudaFree(d_pixels);
    cudaFree(d_output);
}
