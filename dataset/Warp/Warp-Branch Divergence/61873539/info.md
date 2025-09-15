url
https://stackoverflow.com/questions/61873539/cuda-speed-slower-than-expected-image-processing
notes
nvcc -o test test.cu
nsys profile --stats=true ./test