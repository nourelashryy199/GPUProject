#include "openmp.h"
#include <omp.h>
#include <math.h>
#include<string.h>
#include <stdlib.h>
#include <stdint.h>


uint64_t openmp_countGliders(const unsigned char* cells, const size_t width, const size_t height) {
    static const unsigned char GLIDER_1[3][3] =
    { {1, 0, 1},
     {1, 1, 0},
     {0, 0, 0} };
    static const unsigned char GLIDER_2[3][3] =
    { {0, 1, 0},
     {1, 0, 0},
     {1, 0, 1} };
    const unsigned char** cell_index = (const unsigned char**)malloc(sizeof(unsigned char*) * height);
    cell_index[0] = cells;
    for (size_t i = 1; i < height; ++i) {
        cell_index[i] = cell_index[i - 1] + width;
    }
    const size_t width2 = width - 2;
    const size_t height2 = height - 2;
    size_t x;
    uint64_t count = 0;
#pragma omp parallel
    {
#pragma omp for reduction(+:count) schedule(static)
        for (x = 0; x < width2; ++x) {
            for (size_t y = 0; y < height2; ++y) {
                for (unsigned char reflect = 0; reflect <= 1; ++reflect) {
                    for (unsigned char rotate = 0; rotate <= 4; ++rotate) {
                        // x & y
                        for (int gx = 0; gx < 3; ++gx) {
                            for (int gy = 0; gy < 3; ++gy) {
                                int glider_x = reflect ? 2 - gx : gx;
                                int glider_y = gy;
                                for (int i = 0; i < rotate; ++i) {
                                    const int temp_x = glider_x;
                                    glider_x = glider_y;
                                    glider_y = 2 - temp_x;
                                }
                                if (!((cell_index[y + gy][x + gx] && GLIDER_1[glider_y][glider_x]) || 
                                    !(cell_index[y + gy][x + gx] || GLIDER_1[glider_y][glider_x]))) { 
                                    goto fail_group1;
                                }
                            }
                        }
                        ++count;
                        goto pass_group;
                    fail_group1:;
                    }
                }
                for (unsigned char reflect = 0; reflect <= 1; ++reflect) {
                    for (unsigned char rotate = 0; rotate <= 4; ++rotate) {
                        for (int gx = 0; gx < 3; ++gx) {
                            for (int gy = 0; gy < 3; ++gy) {
                                // Calc reflection
                                int glider_x = reflect ? 2 - gx : gx;
                                int glider_y = gy;
                                // Apply rotation
                                for (int i = 0; i < rotate; ++i) {
                                    const int temp_x = glider_x;
                                    glider_x = glider_y;
                                    glider_y = 2 - temp_x;
                                }
                                if (!((cell_index[y + gy][x + gx] && GLIDER_2[glider_y][glider_x]) || // Both are on
                                    !(cell_index[y + gy][x + gx] || GLIDER_2[glider_y][glider_x]))) { // Both are off
                                    // Break from the nested loop
                                    goto fail_group2;
                                }
                            }
                        }
                        // Didn't break early, so we found a glider!
                        ++count;
                        // Continue an outer loop
                        goto pass_group;
                    fail_group2:;
                    }
                }
            pass_group:;
            }
        }
    }
    
    free(cell_index);
    return count;
}
size_t openmp_histogram(const int* numbers, size_t length, int bin_width, int* output) {
    const size_t HISTOGRAM_LEN =
        (size_t)ceil(HISTOGRAM_MAX_VALUE / (double)bin_width);

    memset(output, 0, HISTOGRAM_LEN * sizeof(int)); 

    int num_threads = omp_get_max_threads();

    int* all_hists = calloc((size_t)num_threads * HISTOGRAM_LEN, sizeof(int));
    if (all_hists == NULL) {
        return 0;
    }
    long i; 
    long n = (long)length;

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int* local_hist = all_hists + (size_t)tid * HISTOGRAM_LEN;

    #pragma omp for
        for (i = 0; i < n; ++i) {
            size_t bin = (size_t)(numbers[i] / bin_width);
            ++local_hist[bin];
        }
    }

    for (int t = 0; t < num_threads; ++t) {
        int* local_hist = all_hists + (size_t)t * HISTOGRAM_LEN;
        for (size_t b = 0; b < HISTOGRAM_LEN; ++b) {
            output[b] += local_hist[b];
        }
    }

    free(all_hists);
    return HISTOGRAM_LEN;
}

void openmp_emboss(const unsigned char *pixels, const size_t width, const size_t height, unsigned char* output) {
    static const float EMBOSS_KERNEL[3][3] =
    {
        {-2, -1, 0},
        {-1,  0, 1},
        { 0,  1, 2}
    };

    const unsigned int OUT_WIDTH = width - 2;
    const unsigned int OUT_HEIGHT = height - 2;

    #pragma omp parallel
    {
        int y;
        int x;
        int outH = (int)OUT_HEIGHT;
        int outW = (int)OUT_WIDTH;
#pragma omp for collapse(2)
        for (y = 0; y < outH; ++y) {
            for (x = 0; x < outW; ++x) {
                float pixel_sum = 0.0f;

                for (int ky = 0; ky < 3; ++ky) {
                    for (int kx = 0; kx < 3; ++kx) {
                        const unsigned int offset = (width * (y + ky) + (x + kx)) * 3;

                        const unsigned char R = pixels[offset + 0];
                        const unsigned char G = pixels[offset + 1];
                        const unsigned char B = pixels[offset + 2];

                        const float grey_pixel =
                            (0.2126f * R) + (0.7152f * G) + (0.0722f * B);

                        pixel_sum += grey_pixel * EMBOSS_KERNEL[ky][kx];
                    }
                }

                pixel_sum += 128.0f;
                pixel_sum = pixel_sum < 0.0f ? 0.0f : pixel_sum;
                pixel_sum = pixel_sum > 255.0f ? 255.0f : pixel_sum;

                const unsigned int out_offset = ((unsigned int)y * OUT_WIDTH) + (unsigned int)x;
                output[out_offset] = (unsigned char)pixel_sum;
            }
        }
    }
    
}
