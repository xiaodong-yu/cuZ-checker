#ifndef CUB_DER_H
#define CUB_DER_H

__global__ void derivatives(float *data, float *der, int r3, int r2, int r1, size_t order); 
__global__ void auto_corr(float *data, float *autocor, int r3, int r2, int r1, float avg, size_t autosize); 

#endif /* ----- #ifndef CUB_DER_H  ----- */
