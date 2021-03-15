// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR
#include <stdio.h>
#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/block_reduce.cuh>
#include <cub/util_allocator.cuh>
#include <cub/device/device_reduce.cuh>
#include "CUB_der.h"
#include "CUB_ssim.h"
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

    // Cleanup
    if (d_in) cudaFree(d_in);
    if (d_out) cudaFree(d_out);
    if (d_elapsed) cudaFree(d_elapsed);
}

double grid_sum(float *data, size_t ne){

    TimingGPU timer_GPU;
    // Allocate problem device arrays
    float *d_in = NULL;
    g_allocator.DeviceAllocate((void**)&d_in, sizeof(float) * ne);

    cudaMemcpy(d_in, data, sizeof(float) * ne, cudaMemcpyHostToDevice); 

    timer_GPU.StartCounter();
    // Allocate device output array
    float *d_out = NULL;
    g_allocator.DeviceAllocate((void**)&d_out, sizeof(float) * 1);

    // Request and allocate temporary storage
    void   *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, ne);
    g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes);

    // Run
    DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, ne);
    double duration = timer_GPU.GetCounter();
    printf("GPU CUB sum time: %f ms\n", duration);

    float *h_out = (float*) malloc(sizeof(float) * 1);
    cudaMemcpy(h_out, d_out, sizeof(float) * 1, cudaMemcpyDeviceToHost); 
    printf("test:%e\n", h_out);
    if (d_in) g_allocator.DeviceFree(d_in);
    if (d_out) g_allocator.DeviceFree(d_out);
    if (d_temp_storage) g_allocator.DeviceFree(d_temp_storage);

    return duration;
}

double grid_min(float *data, size_t ne){

    TimingGPU timer_GPU;
    // Allocate problem device arrays
    float *d_in = NULL;
    g_allocator.DeviceAllocate((void**)&d_in, sizeof(float) * ne);

    cudaMemcpy(d_in, data, sizeof(float) * ne, cudaMemcpyHostToDevice); 

    timer_GPU.StartCounter();
    // Allocate device output array
    float *d_out = NULL;
    g_allocator.DeviceAllocate((void**)&d_out, sizeof(float) * 1);

    // Request and allocate temporary storage
    void   *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    DeviceReduce::Min(d_temp_storage, temp_storage_bytes, d_in, d_out, ne);
    g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes);

    // Run
    DeviceReduce::Min(d_temp_storage, temp_storage_bytes, d_in, d_out, ne);
    double duration = timer_GPU.GetCounter();
    printf("GPU CUB min time: %f ms\n", duration);

    float *h_out = (float*) malloc(sizeof(float) * 1);
    cudaMemcpy(h_out, d_out, sizeof(float) * 1, cudaMemcpyDeviceToHost); 
    printf("test:%e\n", h_out);
    if (d_in) g_allocator.DeviceFree(d_in);
    if (d_out) g_allocator.DeviceFree(d_out);
    if (d_temp_storage) g_allocator.DeviceFree(d_temp_storage);
    
    return duration;
}

double grid_max(float *data, size_t ne){

    TimingGPU timer_GPU;
    // Allocate problem device arrays
    float *d_in = NULL;
    g_allocator.DeviceAllocate((void**)&d_in, sizeof(float) * ne);

    cudaMemcpy(d_in, data, sizeof(float) * ne, cudaMemcpyHostToDevice); 

    timer_GPU.StartCounter();
    // Allocate device output array
    float *d_out = NULL;
    g_allocator.DeviceAllocate((void**)&d_out, sizeof(float) * 1);

    // Request and allocate temporary storage
    void   *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_in, d_out, ne);
    g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes);

    // Run
    DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_in, d_out, ne);
    double duration = timer_GPU.GetCounter();
    printf("GPU CUB max time: %f ms\n", duration);

    float *h_out = (float*) malloc(sizeof(float) * 1);
    cudaMemcpy(h_out, d_out, sizeof(float) * 1, cudaMemcpyDeviceToHost); 
    printf("test:%e\n", h_out);
    if (d_in) g_allocator.DeviceFree(d_in);
    if (d_out) g_allocator.DeviceFree(d_out);
    if (d_temp_storage) g_allocator.DeviceFree(d_temp_storage);
    
    return duration;
}

float *Der(float *ddata, float *der, size_t r3, size_t r2, size_t r1, size_t order){

    TimingGPU timer_GPU;
    float *dder;
    const int dsize = (r3-order*2) * (r2-order*2) * (r1-order*2) * sizeof(float);

    cudaMalloc((void**)&dder, dsize); 
    cudaMemcpy(dder, der, dsize, cudaMemcpyHostToDevice); 

    int blksize = (r3-order*2)/(16-order*2)+((r3-order*2)%(16-order*2)?1:0);
    timer_GPU.StartCounter();
    dim3 dimBlock(16, 16);
    dim3 dimGrid(blksize, 1);
    derivatives<<<dimGrid, dimBlock>>>(ddata, dder, r3, r2, r1, order);

    cudaMemcpy(der, dder, dsize, cudaMemcpyDeviceToHost); 

    printf("GPU derivative time: %f ms\n", timer_GPU.GetCounter());
    //for (int i=0;i<(r3-4)*(r2-4)*(r1-4);i++){
    //    if (der[i]!=0.0) printf("ddata%i=%e\n",i,der[i]);
    //}

    cudaFree(dder);

    return der;
}

float *autoCorr(float *ddata, size_t r3, size_t r2, size_t r1, float avg, size_t autosize){

    TimingGPU timer_GPU;
    float *autocor, *dautocor;
    const int dsize = (r3-autosize) * (r2-autosize) * (r1-autosize) * sizeof(float);
    int blksize = (r3-autosize)/(16-autosize)+((r3-autosize)%(16-autosize)?1:0);
    int corsize = blksize * autosize * sizeof(float);

    autocor = (float*)malloc(corsize);
    memset(autocor, 0, corsize);
    cudaMalloc((void**)&dautocor, corsize); 
    cudaMemcpy(dautocor, autocor, corsize, cudaMemcpyHostToDevice);

    timer_GPU.StartCounter();
    dim3 dimBlock(16, 16);
    dim3 dimGrid(blksize, 1);
    const int sMemsize = (16 * dimBlock.x * dimBlock.y + dimBlock.y * autosize) * sizeof(double);
    auto_corr<<<dimGrid, dimBlock, sMemsize>>>(ddata, dautocor, r3, r2, r1, avg, autosize);

    cudaMemcpy(autocor, dautocor, corsize, cudaMemcpyDeviceToHost);

    for (int i=0; i<autosize; i++)
        for (int j=1; j<blksize; j++)
            autocor[blksize*i] += autocor[blksize*i+j];

    printf("GPU autocorr time: %f ms\n", timer_GPU.GetCounter());
    //for (int i=0;i<(r3-4)*(r2-4)*(r1-4);i++){
    //    if (der[i]!=0.0) printf("ddata%i=%e\n",i,der[i]);
    //}

    cudaFree(dautocor);

    return autocor;
}

int SSIM(float *data1, float *data2, size_t r3, size_t r2, size_t r1, int ssimSize, int ssimShift)
{
    TimingGPU timer_GPU;
    float data[246];
    for (int i=0; i<246; i++){
        data[i] = 1;
    }
    int blksize = (r1 - ssimSize) / ssimShift + 1;
    int xsize = ((r2 - ssimSize) / ssimShift + 1)*((r3 - ssimSize) / ssimShift + 1);

    double results[blksize] = { 0 };
    //printf("test=%f, %f\n", data[32], results[32]);

    float *ddata1, *ddata2;
    double *dresults;
    //for (int i=r1*r2*6+r2*6;i<r1*r2*6+r2*6+7;i++){
    ////for (int i=0;i<r1*r2*r3;i++){
    //    printf("data%i=%e, %e\n",i, data1[i], data2[i]);
    //    printf("data%i=%e, %e\n",i, data1[i], data2[i]);

    //}

    const int csize = r3 * r2 * r1 * sizeof(float);
    const int isize = blksize * sizeof(double);

    timer_GPU.StartCounter();
    cudaMalloc((void**)&ddata1,   csize); 
    cudaMalloc((void**)&ddata2,   csize); 
    cudaMalloc((void**)&dresults, isize); 
    cudaMemcpy(ddata1,   data1,   csize, cudaMemcpyHostToDevice); 
    cudaMemcpy(ddata2,   data2,   csize, cudaMemcpyHostToDevice); 
    cudaMemcpy(dresults, results, isize, cudaMemcpyHostToDevice); 

    dim3 dimBlock(64, 1);
    dim3 dimGrid(blksize, 1);
    ssim<<<dimGrid, dimBlock>>>(ddata1, ddata2, dresults, r3, r2, r1, ssimSize, ssimShift);
    cudaMemcpy(results, dresults, isize, cudaMemcpyDeviceToHost); 
    double x=0;
    for (int i=0; i<blksize; i++)
        x += results[i];

    printf("results=%e\n",x);
    printf("GPU ssim time: %f ms\n", timer_GPU.GetCounter());

    cudaFree(ddata1);
    cudaFree(ddata2);
    cudaFree(dresults);

    return 0;
}
