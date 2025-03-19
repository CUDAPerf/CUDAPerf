url
https://stackoverflow.com/questions/64664431/cuda-cublas-issues-solving-many-3x3-dense-linear-systems
notes
nvcc -o before test.cu -lcublas -DBEFORE (performance problem code)
nvcc -o after test.cu  -lcublas (fixed code)
nsys profile --stats=true ./before
nsys profile --stats=true ./after