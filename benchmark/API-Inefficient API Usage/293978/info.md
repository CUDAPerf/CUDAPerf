url
https://forums.developer.nvidia.com/t/why-write-pinned-memory-is-much-slower-than-load-from-pinned-memory-on-multiprocessing-multi-gpu/293978
notes
nvcc -o test test.cu
nsys profile --stats=true ./test