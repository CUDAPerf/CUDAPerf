url
https://stackoverflow.com/questions/64664431/cuda-cublas-issues-solving-many-3x3-dense-linear-systems
notes
nvcc -o buggy buggy.cu -lcublas (performance problem code)
nvcc -o fixed fixed.cu  -lcublas (fixed code)
nsys profile --stats=true ./buggy
nsys profile --stats=true ./fixed