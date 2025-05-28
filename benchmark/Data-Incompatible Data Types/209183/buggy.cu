#include <stdio.h>
#include <stdlib.h>

#define LEN          (1024*8)
#define THREADS      (128)
#define USE_OWN_CVT  (0)

// Macro to catch CUDA errors in CUDA runtime calls
#define CUDA_SAFE_CALL(call)                                          \
do {                                                                  \
    cudaError_t err = call;                                           \
    if (cudaSuccess != err) {                                         \
        fprintf (stderr, "Cuda error in file '%s' in line %i : %s.\n",\
                 __FILE__, __LINE__, cudaGetErrorString(err) );       \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
} while (0)

// Macro to catch CUDA errors in kernel launches
#define CHECK_LAUNCH_ERROR()                                          \
do {                                                                  \
    /* Check synchronous errors, i.e. pre-launch */                   \
    cudaError_t err = cudaGetLastError();                             \
    if (cudaSuccess != err) {                                         \
        fprintf (stderr, "Cuda error in file '%s' in line %i : %s.\n",\
                 __FILE__, __LINE__, cudaGetErrorString(err) );       \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
    /* Check asynchronous errors, i.e. kernel failed (ULF) */         \
    err = cudaDeviceSynchronize();                                    \
    if (cudaSuccess != err) {                                         \
        fprintf (stderr, "Cuda error in file '%s' in line %i : %s.\n",\
                 __FILE__, __LINE__, cudaGetErrorString( err) );      \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
} while (0)

__forceinline__ __device__ float rcp_approx_gpu (float divisor)
{
    float r;
    asm ("rcp.approx.ftz.f32 %0,%1;\n\t" : "=f"(r) : "f"(divisor));
    return r;
}

/* integer division for dividend and divisor < 100,000 */
__device__ unsigned int udiv_lt_100000_nouse (unsigned int x, unsigned int y)
{
    const float magic = 1.2582912e+7f; // 0x1.8p+23
    const unsigned int magic_i = __float_as_int (magic);


    float divisor = (float)y;
    float dividend = (float)x;

    float t = rcp_approx_gpu (divisor);
    t = __int_as_float (__float_as_int (t) + 1); // make sure quotient is never too big
    t = t * dividend;

    unsigned int q = (unsigned int)t;
    return q;
}


__global__ void divtest2 (unsigned int *result, int len)
{
    int stride = gridDim.x * blockDim.x;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    for (int i = tid; i < len; i += stride) {
        unsigned int dividend = i;
        unsigned int flag = 0;
        for (unsigned int divisor = 1; divisor < LEN; divisor++) {
            unsigned int ref = dividend / divisor;
            unsigned int res = udiv_lt_100000_nouse (dividend, divisor);
            flag += res != ref;
        }
        result [i] = flag;
    }
}

int test (void)
{
    dim3 dimBlock(THREADS);
    int threadBlocks = (LEN + (dimBlock.x - 1)) / dimBlock.x;
    dim3 dimGrid(threadBlocks);
    unsigned int *r = 0, *d_r = 0;

    r = (unsigned int *) malloc (sizeof (r[0]) * LEN);
    CUDA_SAFE_CALL (cudaMalloc((void**)&d_r, sizeof(d_r[0]) * LEN));
    CUDA_SAFE_CALL (cudaMemset(d_r, 0xff, sizeof(d_r[0]) * LEN)); // all ones
    divtest2<<<dimGrid,dimBlock>>>(d_r, LEN);
    CHECK_LAUNCH_ERROR();
    CUDA_SAFE_CALL (cudaMemcpy (r, d_r, sizeof (r[0]) * LEN, cudaMemcpyDeviceToHost));

    for (int i = 0; i < LEN; i++) {
        if (r[i] == 0xffffffff) {
            printf ("counter not written i=%d\n", i);
            return EXIT_FAILURE;
        }
        if (r[i] != 0) {
            printf ("division failures i=%d  r=%u\n", i, r[i]);
            return EXIT_FAILURE;
        }
    }
    CUDA_SAFE_CALL (cudaFree (d_r));
    free (r);
    return EXIT_SUCCESS;
}

int main(){
    //cudaSetDevice(1);
    for(int i=0;i<10;i++){
        test();
        printf("test %d\n",i);
    }
}