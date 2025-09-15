#include <cuda.h>
#include <stdio.h>
#include <iostream>

#define mul 1


__global__ 
void test_float(short* in, float* out, int a, int b, int c, int d, int e)
{
	int pixel = threadIdx.x + blockIdx.x * blockDim.x;
	
	if(pixel < mul * a * b * d * e)
	{
		float tmp = 0.0;
		int id = 0;
		float tmp2;
		
		for(int i = 0; i < c; i++)
		{
			tmp2 = (float)in[id];
			//tmp = tmp + 3.32 + tmp2;//SLOW
			tmp = tmp + (float)3.32 + tmp2;//FAST
			id += 328;
		}

		
		out[pixel] = tmp/(float)c;
	}
}

int test()
{

	int a = 100;
	int b = 10;
	int c = 128;
	int d = 321;
	int e = 161;
	
	
	float* d_out;
	float* h_out;
	short* d_in;
	short* h_in;
	
	h_out = (float*)malloc(mul * a * b * d * e * sizeof(float));
	h_in = (short*)malloc(a * b * 1000 * c * sizeof(short));//normally contains data
	cudaMalloc(&d_out, mul * a * b * d * e * sizeof(float));
	cudaMalloc(&d_in, a * b * 1000 * c * sizeof(short));
	cudaMemcpy(d_in, h_in, a * b * 1000 * c * sizeof(short), cudaMemcpyHostToDevice);


	dim3 blocks((int)ceil( (float)(mul * a * b * d * e) / (float)1024));
	dim3 threads_per_block(1024);
	
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start); 
    test_float<<<blocks, threads_per_block>>>(d_in, d_out, a, b, c, d, e);
    cudaEventRecord(stop); 
    cudaEventSynchronize(stop); 
    float elapsed_time_float;
    cudaEventElapsedTime(&elapsed_time_float, start, stop); 
    std::cout << "test_float kernel execution time: " << elapsed_time_float << " ms" << std::endl;

    
    cudaMemcpy(h_out, d_out, a * b * d * e * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 10032; i < 10035; i++) {
        printf("%f\n", h_out[i]);
    }

    
    free(h_out);
    free(h_in);
    cudaFree(d_out);
    cudaFree(d_in);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

int main(){
	//cudaSetDevice(1);
	for(int i=0;i<10;i++){
		test();
	}
}