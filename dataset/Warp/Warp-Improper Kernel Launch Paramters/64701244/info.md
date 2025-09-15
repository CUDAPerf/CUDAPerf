url
https://stackoverflow.com/questions/64701244/explain-gpu-nvidia-execution-time
notes
nvcc -o test test.cu
nsys profile --stats=true ./test