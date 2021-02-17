// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR
#include <stdio.h>
#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/block_reduce.cuh>
#include <cub/util_allocator.cuh>
#include <cub/device/device_reduce.cuh>
#include "timingGPU.h"

using namespace cub;

//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------
/// Timing iterations
int g_timing_iterations = 100;

/// Default grid size
int g_grid_size = 1;

CachingDeviceAllocator  g_allocator(true);  // Caching allocator for device memory

template <
    int                     BLOCK_THREADS,
    int                     ITEMS_PER_THREAD,
    BlockReduceAlgorithm    ALGORITHM>
__global__ void BlockSumKernel(
    float       *d_in,          // Tile of input
    float       *d_out,         // Tile aggregate
    clock_t     *d_elapsed)     // Elapsed cycle count of block reduction
{
    // Specialize BlockReduce type for our thread block
    typedef BlockReduce<float, BLOCK_THREADS, ALGORITHM> BlockReduceT;

    // Shared memory
    __shared__ typename BlockReduceT::TempStorage temp_storage;

    // Per-thread tile data
    float data[ITEMS_PER_THREAD];
    LoadDirectStriped<BLOCK_THREADS>(threadIdx.x, d_in, data);

    // Start cycle timer
    clock_t start = clock();

    // Compute sum
    float aggregate = BlockReduceT(temp_storage).Sum(data);

    // Stop cycle timer
    clock_t stop = clock();

    // Store aggregate and elapsed clocks
    if (threadIdx.x == 0)
    {
        *d_elapsed = (start > stop) ? start - stop : stop - start;
        *d_out = aggregate;
    }
}

/**
 * Test thread block reduction
 */
void block_reduce(float *data1, float *data2, double *ddiff, int fsize, double *absErrPDF, double *results, size_t r3, size_t r2, size_t r1, size_t ne)
{
    TimingGPU timer_GPU;
    const int BLOCK_THREADS = 1024;
    const int ITEMS_PER_THREAD = 4;
    const int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;

    // Initialize device arrays
    float *d_in         = NULL;
    float *d_out        = NULL;
    clock_t *d_elapsed  = NULL;
    cudaMalloc((void**)&d_in,          sizeof(float) * ne);
    cudaMalloc((void**)&d_out,         sizeof(float) * 1);
    cudaMalloc((void**)&d_elapsed,     sizeof(clock_t));

    // Kernel props
    int max_sm_occupancy;
    MaxSmOccupancy(max_sm_occupancy, BlockSumKernel<BLOCK_THREADS, ITEMS_PER_THREAD, BLOCK_REDUCE_WARP_REDUCTIONS>, BLOCK_THREADS);

    // Copy problem to device
    cudaMemcpy(d_in, data1, sizeof(float) * ne, cudaMemcpyHostToDevice);

    printf("BlockReduce algorithm on %d items (%d timing iterations, %d blocks, %d threads, %d items per thread, %d SM occupancy):\n",
        TILE_SIZE, g_timing_iterations, g_grid_size, BLOCK_THREADS, ITEMS_PER_THREAD, max_sm_occupancy);

    timer_GPU.StartCounter();
    // Run aggregate/prefix kernel
    BlockSumKernel<BLOCK_THREADS, ITEMS_PER_THREAD, BLOCK_REDUCE_WARP_REDUCTIONS><<<g_grid_size, BLOCK_THREADS>>>(
        d_in,
        d_out,
        d_elapsed);

    // Check for kernel errors and STDIO from the kernel, if any
    cudaPeekAtLastError();
    cudaDeviceSynchronize();
    printf("GPU timing: %f ms\n", timer_GPU.GetCounter());

    float *h_out = (float*) malloc(sizeof(float) * 1);
    cudaMemcpy(h_out, d_out, sizeof(float) * 1, cudaMemcpyDeviceToHost); 
    printf("test:%e\n", h_out);

    // Cleanup
    if (d_in) cudaFree(d_in);
    if (d_out) cudaFree(d_out);
    if (d_elapsed) cudaFree(d_elapsed);
}

void grid_reduce(float *data1, float *data2, double *ddiff, int fsize, double *absErrPDF, double *results, size_t r3, size_t r2, size_t r1, size_t ne){

    TimingGPU timer_GPU;
    // Allocate problem device arrays
    float *d_in = NULL;
    g_allocator.DeviceAllocate((void**)&d_in, sizeof(float) * ne);

    cudaMemcpy(d_in, data2, sizeof(float) * ne, cudaMemcpyHostToDevice); 
    // Allocate device output array
    float *d_out = NULL;
    g_allocator.DeviceAllocate((void**)&d_out, sizeof(float) * 1);

    timer_GPU.StartCounter();
    // Request and allocate temporary storage
    void   *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, ne);
    g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes);

    // Run
    DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, ne);
    printf("GPU timing: %f ms\n", timer_GPU.GetCounter());

    float *h_out = (float*) malloc(sizeof(float) * 1);
    cudaMemcpy(h_out, d_out, sizeof(float) * 1, cudaMemcpyDeviceToHost); 
    printf("test:%e\n", h_out);
}

