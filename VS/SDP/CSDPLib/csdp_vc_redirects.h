#pragma once

#include <string.h>
#include "mkl.h"

#define strcasecmp _stricmp

#define ddot_(np, x, incxp, y, incyp) cblas_ddot(*np, x, *incxp, y, *incyp)

/* 
* "The BLAS routines follow the Fortran convention of storing two-dimensional arrays using column-major layout."
* from: https://software.intel.com/content/www/us/en/develop/documentation/mkl-developer-reference-c/top/blas-and-sparse-blas-routines/blas-routines/matrix-storage-schemes-for-blas-routines.html
* 
* See also [sdp.c]: "All "2-d" arrays are stored in Fortran style, column major order"
*/

//#define MKL_CBLAS_LAYOUT_FOR_CSDP   CblasRowMajor
#define MKL_CBLAS_LAYOUT_FOR_CSDP   CblasColMajor

#pragma warning( push )
#pragma warning ( disable: 26812 )

static inline CBLAS_TRANSPOSE parse_transpose_character(char transpose)
{
    CBLAS_TRANSPOSE trans_ = CblasNoTrans;

    switch (transpose)
    {
    case 'N':
    case 'n':
        trans_ = CblasNoTrans;
        break;

    case 'T':
    case 't':
        trans_ = CblasTrans;
        break;

    default:
        break;
    }

    return trans_;
}

static inline CBLAS_UPLO parse_upper_or_lower_character(const char uplo)
{
    CBLAS_UPLO uplo_ = CblasUpper;

    switch (uplo)
    {
    case 'U':
    case 'u':
        uplo_ = CblasUpper;
        break;

    case 'L':
    case 'l':
        uplo_ = CblasLower;
        break;

    default:
        break;
    }

    return uplo_;
}

static inline CBLAS_DIAG parse_diag_character(const char diag)
{
    CBLAS_DIAG diag_ = CblasNonUnit;

    switch (diag)
    {
    case 'N':
    case 'n':
        diag_ = CblasNonUnit;
        break;

    case 'U':
    case 'u':
        diag_ = CblasUnit;
        break;

    default:
        break;
    }

    return diag_;
}

#pragma warning ( pop )

/*
* From: https://software.intel.com/content/www/us/en/develop/documentation/mkl-developer-reference-c/top/blas-and-sparse-blas-routines/blas-routines/blas-level-2-routines/cblas-gemv.html
* 
void cblas_dgemv (
    const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE trans, 
    const MKL_INT m, const MKL_INT n, 
    const double alpha, const double *a, 
    const MKL_INT lda, const double *x, 
    const MKL_INT incx, const double beta, double *y, 
    const MKL_INT incy);

From: https://www.ibm.com/support/knowledgecenter/SSGH2K_13.1.2/com.ibm.xlc131.aix.doc/proguide/blas_syntax.html
void dgemv(
    const char *trans, 
    int *m, int *n, 
    double *alpha, void *a, 
    int *lda, void *x, 
    int *incx, double *beta, void *y, 
    int *incy);
*/

static inline void dgemv_(
    const char* trans,
    int* m, int* n,
    double* alpha, void* a,
    int* lda, void* x,
    int* incx, double* beta, void* y,
    int* incy)
{
    MKL_INT m_ = *m;
    MKL_INT n_ = *n;
    double alpha_ = *alpha;
    double* a_ = (double*)a;
    MKL_INT lda_ = *lda;
    MKL_INT incx_ = *incx;
    double* x_ = (double*)x;
    double beta_ = *beta;
    double* y_ = (double*)y;
    MKL_INT incy_ = *incy;
    CBLAS_TRANSPOSE trans_ = parse_transpose_character(*trans);

    cblas_dgemv(
        MKL_CBLAS_LAYOUT_FOR_CSDP, 
        trans_, m_, n_, alpha_, a_, lda_, x_, incx_, beta_, y_, incy_);
}

/*
* From: https://software.intel.com/content/www/us/en/develop/documentation/mkl-developer-reference-c/top/blas-and-sparse-blas-routines/blas-routines/blas-level-3-routines/cblas-gemm.html
* void cblas_dgemm (
    const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE transa, const CBLAS_TRANSPOSE transb, 
    const MKL_INT m, const MKL_INT n, const MKL_INT k, const double alpha, 
    const double *a, const MKL_INT lda, 
    const double *b, const MKL_INT ldb, 
    const double beta, 
    double *c, const MKL_INT ldc);
*
* Implements: C := alpha*op(A)*op(B) + beta*C
* 
* m: Specifies the number of rows of the matrix op(A) and of the matrix C. The value of m must be at least zero.
* n: Specifies the number of columns of the matrix op(B) and the number of columns of the matrix C. The value of n must be at least zero.
* k: Specifies the number of columns of the matrix op(A) and the number of rows of the matrix op(B). The value of k must be at least zero.

* From: https://www.ibm.com/support/knowledgecenter/SSGH2K_13.1.2/com.ibm.xlc131.aix.doc/proguide/blas_syntax.html
* void dgemm(
    const char *transa, const char *transb,
    int *l, int *n, int *m, double *alpha, 
    const void *a, int *lda, 
    void *b, int *ldb, 
    double *beta, 
    void *c, int *ldc);

* l: represents the number of rows in output matrix c. The number of rows must be greater than or equal to zero, and less than the leading dimension of c.
* n: represents the number of columns in output matrix c. The number of columns must be greater than or equal to zero.
* m: represents:
        the number of columns in matrix a, if 'N' or 'n' is used for the transa parameter
        the number of rows in matrix a, if 'T' or 't' is used for the transa parameter
    and:
        the number of rows in matrix b, if 'N' or 'n' is used for the transb parameter
        the number of columns in matrix b, if 'T' or 't' is used for the transb parameter
    m must be greater than or equal to zero.
*/

static inline void dgemm_(
    const char* transa, const char* transb,
    int* l, int* n, int* m, double* alpha,
    const void* a, int* lda,
    void* b, int* ldb,
    double* beta,
    void* c, int* ldc)
{
    MKL_INT m_ = *l;
    MKL_INT n_ = *n;
    MKL_INT k_ = *m;
    double alpha_ = *alpha;
    const double* a_ = (double*)a;
    MKL_INT lda_ = *lda;
    double* b_ = (double*)b;
    MKL_INT ldb_ = *ldb;
    double beta_ = *beta;
    double* c_ = (double*)c;
    MKL_INT ldc_ = *ldc;
    
    CBLAS_TRANSPOSE transa_ = parse_transpose_character(*transa);
    CBLAS_TRANSPOSE transb_ = parse_transpose_character(*transb);

    cblas_dgemm(MKL_CBLAS_LAYOUT_FOR_CSDP,
        transa_, transb_,
        m_, n_, k_,
        alpha_, a_,
        lda_, b_, ldb_, beta_, 
        c_, ldc_
    );
}

/*
* From: https://software.intel.com/content/www/us/en/develop/documentation/mkl-developer-reference-c/top/blas-and-sparse-blas-routines/blas-routines/blas-level-2-routines/cblas-symv.html
* void cblas_dsymv (
    const CBLAS_LAYOUT Layout, 
    const CBLAS_UPLO uplo, const MKL_INT n, 
    const double alpha, const double *a, const MKL_INT lda, 
    const double *x, const MKL_INT incx, 
    const double beta, double *y, const MKL_INT incy);

*
* From: https://www.cs.kent.ac.uk/projects/cxxr/APIdoc/html/BLAS_8h_source.html
* void  dsymv(
    const char *uplo, const int *n, 
    const double *alpha, const double *a, const int *lda,
    const double *x, const int *incx,
    const double *beta, double *y, const int *incy);
*/

static inline void  dsymv_(
    const char* uplo, const int* n,
    const double* alpha, const double* a, const int* lda,
    const double* x, const int* incx,
    const double* beta, double* y, const int* incy)
{
    MKL_INT n_ = *n;
    double alpha_ = *alpha;
    double* a_ = (double*)a;
    MKL_INT lda_ = *lda;
    double* x_ = (double*)x;
    MKL_INT incx_ = *incx;
    double beta_ = *beta;
    double* y_ = (double*)y;
    MKL_INT incy_ = *incy;

    CBLAS_UPLO uplo_ = parse_upper_or_lower_character(*uplo);

    cblas_dsymv(
        MKL_CBLAS_LAYOUT_FOR_CSDP,
        uplo_, n_, alpha_, a_, lda_,
        x_, incx_, beta_, y_, incy_
    );
}

/*
* From: https://software.intel.com/content/www/us/en/develop/documentation/mkl-developer-reference-c/top/blas-and-sparse-blas-routines/blas-routines/blas-level-2-routines/cblas-trmv.html
* void cblas_dtrmv (
    const CBLAS_LAYOUT Layout, 
    const CBLAS_UPLO uplo, const CBLAS_TRANSPOSE trans, const CBLAS_DIAG diag, 
    const MKL_INT n, const double *a, const MKL_INT lda, 
    double *x, const MKL_INT incx);
* 
* From: 
* void dtrmv(
    const char *uplo, const char *trans, const char *diag,
    const int *n, const double *a, const int *lda,
    double *x, const int *incx);
*/

static inline void dtrmv_(
    const char* uplo, const char* trans, const char* diag,
    const int* n, const double* a, const int* lda,
    double* x, const int* incx)
{
    CBLAS_UPLO uplo_ = parse_upper_or_lower_character(*uplo);
    CBLAS_TRANSPOSE trans_ = parse_transpose_character(*trans);
    CBLAS_DIAG diag_ = parse_diag_character(*diag);
    MKL_INT n_ = *n;
    double* a_ = (double*)a;
    MKL_INT lda_ = *lda;
    double* x_ = (double*)x;
    MKL_INT incx_ = *incx;

    cblas_dtrmv(
        MKL_CBLAS_LAYOUT_FOR_CSDP,
        uplo_, trans_, diag_, n_, a_, lda_, x_, incx_);
}

/*
* From: https://software.intel.com/content/www/us/en/develop/documentation/mkl-developer-reference-c/top/blas-and-sparse-blas-routines/blas-routines/blas-level-1-routines-and-functions/cblas-asum.html
* double cblas_dasum (const MKL_INT n, const double *x, const MKL_INT incx);
* 
* From: https://www.cs.kent.ac.uk/projects/cxxr/APIdoc/html/BLAS_8h_source.html
* double dasum(const int* n, const double* dx, const int* incx);
*/

static inline double dasum_(const int* n, const double* dx, const int* incx)
{
    MKL_INT n_ = *n;
    double* dx_ = (double*)dx;
    MKL_INT incx_ = *incx;

    return cblas_dasum(n_, dx_, incx_);
}

/*
* From: https://software.intel.com/content/www/us/en/develop/documentation/mkl-developer-reference-c/top/blas-and-sparse-blas-routines/blas-routines/blas-level-1-routines-and-functions/cblas-i-amax.html
* CBLAS_INDEX cblas_idamax (const MKL_INT n, const double *x, const MKL_INT incx);
* 
* From: https://www.cs.kent.ac.uk/projects/cxxr/APIdoc/html/BLAS_8h_source.html
* int idamax(const int* n, const double* dx, const int* incx);
*/

static inline int idamax_(const int* n, const double* dx, const int* incx)
{
    MKL_INT n_ = *n;
    double* x_ = (double*)dx;
    MKL_INT incx_ = *incx;

    CBLAS_INDEX result = cblas_idamax(n_, x_, incx_);
    return (int)result;
}

/*
* From: https://software.intel.com/content/www/us/en/develop/documentation/mkl-developer-reference-c/top/blas-and-sparse-blas-routines/blas-routines/blas-level-1-routines-and-functions/cblas-nrm2.html
* double cblas_dnrm2 (const MKL_INT n, const double *x, const MKL_INT incx);
* 
* From: https://www.cs.kent.ac.uk/projects/cxxr/APIdoc/html/BLAS_8h_source.html
* double dnrm2(const int* n, const double* dx, const int* incx);
*/

static inline double dnrm2_(const int* n, const double* dx, const int* incx)
{
    MKL_INT n_ = *n;
    double* x_ = (double*)dx;
    MKL_INT incx_ = *incx;

    return cblas_dnrm2(n_, x_, incx_);
}
