
__global__ void k1(int s, int e, float *r){

  float val = 0;
  for (int i = s+threadIdx.x; i < e; i++){
    float x = i;
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
  }
}
