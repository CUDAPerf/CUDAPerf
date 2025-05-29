url
https://forums.developer.nvidia.com/t/tiled-matrix-multiplication-vastly-slower-than-simple-matrix-multiplication/200244
notes
nvcc -o test test.cu
nsys profile --stats=true ./test