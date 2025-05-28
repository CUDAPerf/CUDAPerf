#include <cstdio>
#include <cstdlib>
// the movement
// start:
//  A B
//  C D
// step 1:
//  B A
//  C D
// step 2:
//  C A
//  B D
// step 3:
//  A C
//  B D

__global__ void t_bad(int do_print){

  int u[32];
  for (int i = 0; i < 32; i++) u[i] = threadIdx.x*32+i;
  if (u[0] >= do_print)
    // print data
    for (int i = 0; i < 32; i++)
      if (threadIdx.x == i){
        for (int j = 0; j < 32; j++)  printf("%d ", u[j]);
        printf("\n");}
  #pragma unroll 31
  for (int i = 1; i < 32; i++){
    int idx = threadIdx.x^i;
    u[idx] = __shfl_sync(0xFFFFFFFF, u[idx], idx);}
  if (u[0] >= do_print)
    // print data
    for (int i = 0; i < 32; i++)
      if (threadIdx.x == i){
        for (int j = 0; j < 32; j++)  printf("%d ", u[j]);
        printf("\n");}
}


int main(int argc, char *argv[]){
  cudaSetDevice(1);
  int do_print = 1024;
  if (argc > 1) do_print = atoi(argv[1]);
  int n = 1024*1024;
  for(int i=0;i<10;i++){
    t_bad<<<n,32>>>(do_print);
    cudaDeviceSynchronize();
  }
}