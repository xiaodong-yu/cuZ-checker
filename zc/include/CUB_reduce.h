#ifndef CUB_REDUCE_H
#define CUB_REDUCE_H

void block_reduce(float *data1, float *data2, double *ddiff, int fsize, double *absErrPDF, double *results, size_t r3, size_t r2, size_t r1, size_t ne);

void grid_reduce(float *data1, float *data2, double *ddiff, int fsize, double *absErrPDF, double *results, size_t r3, size_t r2, size_t r1, size_t ne);

#endif /* ----- #ifndef CUB_REDUCE_H  ----- */
