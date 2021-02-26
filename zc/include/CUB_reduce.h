#ifndef CUB_REDUCE_H
#define CUB_REDUCE_H

void block_reduce(float *data1, float *data2, double *ddiff, int fsize, double *absErrPDF, double *results, size_t r3, size_t r2, size_t r1, size_t ne);

void grid_reduce(float *data1, float *data2, double *ddiff, int fsize, double *absErrPDF, double *results, size_t r3, size_t r2, size_t r1, size_t ne);

float *Der(float *ddata, float *der, size_t r3, size_t r2, size_t r1, size_t order);

float *autoCorr(float *ddata, size_t r3, size_t r2, size_t r1, float avg, size_t autosize);

int SSIM(float *data1, float *data2, size_t r3, size_t r2, size_t r1, int ssimSize, int ssimShift);

#endif /* ----- #ifndef CUB_REDUCE_H  ----- */
