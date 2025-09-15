
/* -2**22 <= a < 2**22 */
__device__ float fast_int_to_float (int a)
{
    const float fmagic = (1 << 23) + (1 << 22);
    const int imagic = __float_as_int (fmagic);
    return __int_as_float (imagic + a) - fmagic;
}

__global__ void k1(int s, int e, float *r){

  float val = 0;
  for (int i = s+threadIdx.x; i < e; i++){
    float x = i;
    val += x;}
  r[threadIdx.x] = val;
}

__global__ void k2(int s, int e, float *r){

  float val = 0;
  for (int i = s+threadIdx.x; i < e; i++){
    float x = fast_int_to_float(i);
    val += x;}
  r[threadIdx.x] = val;
}

int main(){
  //cudaSetDevice(1);
  const int nBLK = 58*3;
  const int nTPB = 512;
  const int s = 101;
  const int e = 1024;
  float *r;
  cudaMalloc(&r, nTPB*sizeof(*r));
  for(int i=0;i<10;i++){
    k1<<<nBLK, nTPB>>>(s,e,r);
    cudaDeviceSynchronize();
    k2<<<nBLK, nTPB>>>(s,e,r);
    cudaDeviceSynchronize();
  }
}
