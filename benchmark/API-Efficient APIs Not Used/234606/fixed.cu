#include <iostream>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>


/*
    Ensures safe cuda application executions
*/
#define gpuSafeExec(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

/*
    Clears shared memory which is not full of previous
    numbers. Shmem is remembers values between consecutive
    kernel calls.
*/
__device__ void flushShmem(float *shmem, int shmemSize){
    for (int i = 0; i < shmemSize; i ++)
        shmem[i] = 0.0f;
    return;
}

struct Cluster{
	float* x;
	float* y;
	float* z;
};

void populateClusters(Cluster A, Cluster B, int n) {
	for (int i = 0; i < n; i++) {
		A.x[i] = 1000.0f * (float)rand() / (float)RAND_MAX;
		A.y[i] = 1000.0f * (float)rand() / (float)RAND_MAX;
		A.z[i] = 1000.0f * (float)rand() / (float)RAND_MAX;
		if ((float)rand() / (float)RAND_MAX < 0.01f) {
			B.x[i] = A.x[i] + 10.0f * (float)rand() / (float)RAND_MAX;
			B.y[i] = A.y[i] + 10.0f * (float)rand() / (float)RAND_MAX;
			B.z[i] = A.z[i] + 10.0f * (float)rand() / (float)RAND_MAX;
		}
		else {
            B.x[i] = A.x[i] + 1.0f * (float)rand() / (float)RAND_MAX;
            B.y[i] = A.y[i] + 1.0f * (float)rand() / (float)RAND_MAX;
            B.z[i] = A.z[i] + 1.0f * (float)rand() / (float)RAND_MAX;
		}
	}
}


struct MatrixIndex{
    int i = 0;
    int j = 0;
};

std::ostream& operator<<(std::ostream& os, const MatrixIndex& m){
    os << "(" << m.i << "," << m.j << ")";
    return os;
}

struct ConvertLinearIndexToTriangularMatrixIndex{
    int dim;

    __host__ __device__
    ConvertLinearIndexToTriangularMatrixIndex(int dimension) : dim(dimension){}

    __host__ __device__
    MatrixIndex operator()(int linear) const {
        MatrixIndex result;
       //check if those indices work for you

        //compute i and j from linear index https://stackoverflow.com/questions/27086195/linear-index-upper-triangular-matrix
        result.i = dim - 2 - floor(sqrt(-8*linear + 4*dim*(dim-1)-7)/2.0 - 0.5);
        result.j = linear + result.i + 1 - dim*(dim-1)/2 + (dim-result.i)*((dim-result.i)-1)/2;

        return result;
    }
};

struct ComputeDelta{
    Cluster A;
    Cluster B;

    __host__ __device__
    ComputeDelta(Cluster _A, Cluster _B){
        /* init A and B*/
        A = _A;
        B = _B;
    }

    __host__ __device__
    float operator()(const MatrixIndex& index) const{
        
        float da = 0;
        float db = 0;

        da = sqrt((A.x[index.i]-A.x[index.j])*(A.x[index.i]-A.x[index.j])
                    + (A.y[index.i]-A.y[index.j])*(A.y[index.i]-A.y[index.j])
                    + (A.z[index.i]-A.z[index.j])*(A.z[index.i]-A.z[index.j]));
        db = sqrt((B.x[index.i]-B.x[index.j])*(B.x[index.i]-B.x[index.j])
                    + (B.y[index.i]-B.y[index.j])*(B.y[index.i]-B.y[index.j])
                    + (B.z[index.i]-B.z[index.j])*(B.z[index.i]-B.z[index.j]));

        return (da-db) * (da-db);
    }
};

float solveGPU_thrust(Cluster A, Cluster B, int n) {
    const int dim = n;
    const int elems = round(n*(n-1)/2); //upper triangular(without diagonal) number of elements formula
    auto matrixIndexIterator = thrust::make_transform_iterator(
        thrust::make_counting_iterator(0),
        ConvertLinearIndexToTriangularMatrixIndex{dim}
    );


    //for(int i = 0; i < elems; i++){
    //    std::cout << matrixIndexIterator[i] << " ";
    //}
    
    float result = thrust::transform_reduce(
        matrixIndexIterator, 
        matrixIndexIterator + elems, 
        ComputeDelta{A,B}, 
        float(0), 
        thrust::plus<float>{}
    );
    return sqrt(1/((float)n*((float)n-1)) * result);
}

float solveCPU(Cluster A, Cluster B, int n) {
	float difference = 0.0f;
	for (int i = 0; i < n-1; i++) {
		float tmp = 0.0f;
		for (int j = i+1; j < n; j++) {
			float diff_a = sqrt((A.x[i]-A.x[j])*(A.x[i]-A.x[j])
				+ (A.y[i]-A.y[j])*(A.y[i]-A.y[j])
				+ (A.z[i]-A.z[j])*(A.z[i]-A.z[j]));
			float diff_b = sqrt((B.x[i]-B.x[j])*(B.x[i]-B.x[j])
				+ (B.y[i]-B.y[j])*(B.y[i]-B.y[j])
				+ (B.z[i]-B.z[j])*(B.z[i]-B.z[j]));
			tmp += (diff_a-diff_b) * (diff_a-diff_b);
		}
		difference += tmp;
	}
	return sqrt(1/((float)n*((float)n-1)) * difference);
}

int main(int argc, char **argv){
    constexpr int N = 1024*64;

	Cluster A, B;
	A.x = A.y = A.z = B.x = B.y = B.z = NULL;
	Cluster dA, dB;
	dA.x = dA.y = dA.z = dB.x = dB.y = dB.z = NULL;
	float diff_CPU, diff_GPU;

	// parse command line
	int device = 0;
	if (argc == 2) 
		device = atoi(argv[1]);
	if (cudaSetDevice(device) != cudaSuccess){
		fprintf(stderr, "Cannot set CUDA device!\n");
		exit(1);
	}

	printf("Number of points per cluster: %d\n", N);
	cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);
    printf("Using device %d: \"%s\"\n", device, deviceProp.name);
	//printf("%d \n",*deviceProp.maxGridSize);

	// create events for timing
	cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

	// allocate and set host memory
	A.x = (float*)malloc(N*sizeof(A.x[0]));
	A.y = (float*)malloc(N*sizeof(A.y[0]));
	A.z = (float*)malloc(N*sizeof(A.z[0]));
	B.x = (float*)malloc(N*sizeof(B.x[0]));
    B.y = (float*)malloc(N*sizeof(B.y[0]));
    B.z = (float*)malloc(N*sizeof(B.z[0]));
	populateClusters(A, B, N);      
 
	// allocate and set device memory
	if (cudaMalloc((void**)&dA.x, N*sizeof(dA.x[0])) != cudaSuccess
	|| cudaMalloc((void**)&dA.y, N*sizeof(dA.y[0])) != cudaSuccess
	|| cudaMalloc((void**)&dA.z, N*sizeof(dA.z[0])) != cudaSuccess
	|| cudaMalloc((void**)&dB.x, N*sizeof(dB.x[0])) != cudaSuccess
    || cudaMalloc((void**)&dB.y, N*sizeof(dB.y[0])) != cudaSuccess
    || cudaMalloc((void**)&dB.z, N*sizeof(dB.z[0])) != cudaSuccess) {
		fprintf(stderr, "Device memory allocation error!\n");
		cudaError_t err = cudaGetLastError();
		if (err != cudaSuccess){
			printf("CUDA ERROR while executing the kernel: %s\n",cudaGetErrorString(err));
			return 103;
		}
		goto cleanup;
	}
	cudaMemcpy(dA.x, A.x, N*sizeof(dA.x[0]), cudaMemcpyHostToDevice);
	cudaMemcpy(dA.y, A.y, N*sizeof(dA.y[0]), cudaMemcpyHostToDevice);
	cudaMemcpy(dA.z, A.z, N*sizeof(dA.z[0]), cudaMemcpyHostToDevice);
	cudaMemcpy(dB.x, B.x, N*sizeof(dB.x[0]), cudaMemcpyHostToDevice);
    cudaMemcpy(dB.y, B.y, N*sizeof(dB.y[0]), cudaMemcpyHostToDevice);
    cudaMemcpy(dB.z, B.z, N*sizeof(dB.z[0]), cudaMemcpyHostToDevice);

	// solve on CPU
    printf("Solving on CPU...\n");
	cudaEventRecord(start, 0);
	diff_CPU = solveCPU(A, B, N);
	cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float time;
    cudaEventElapsedTime(&time, start, stop);
    printf("CPU performance: %f megapairs/s\n",
        float(N)*float(N-1)/2.0f/time/1e3f);

	
    // solve on GPU with thrust
	printf("Solving on GPU with thrust...\n");
	cudaEventRecord(start, 0);
	// run it 10x for more accurately timing results
    for (int i = 0; i < 10; i++){
		diff_GPU = solveGPU_thrust(dA, dB, N);

	}
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
	printf("GPU performance with thrust: %f megapairs/s\n",
        float(N)*float(N-1)/2.0f/time/1e2f);
    printf("time:%f\n",time);

	printf("CPU diff: %f\nGPU diff: %f\n", diff_CPU, diff_GPU);
	// check GPU results
	if ( fabsf((diff_CPU-diff_GPU) / ((diff_CPU+diff_GPU)/2.0f)) < 0.01f)
		printf("Test OK :-).\n");
	else
		 fprintf(stderr, "Data mismatch: %f should be %f :-(\n", diff_GPU, diff_CPU);

cleanup:
	cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
	if (dA.x) cudaFree(dA.x);
	if (dA.y) cudaFree(dA.y);
	if (dA.z) cudaFree(dA.z);
	if (dB.x) cudaFree(dB.x);
    if (dB.y) cudaFree(dB.y);
    if (dB.z) cudaFree(dB.z);
	if (A.x) free(A.x);
	if (A.y) free(A.y);
	if (A.z) free(A.z);
	if (B.x) free(B.x);
    if (B.y) free(B.y);
    if (B.z) free(B.z);

	return 0;
}