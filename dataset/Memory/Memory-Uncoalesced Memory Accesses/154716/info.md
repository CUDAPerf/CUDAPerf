url
https://forums.developer.nvidia.com/t/32-byte-coalesced-access-is-faster-than-128-byte-coalesced-access/154716
notes
nvcc -o test test.cu
nsys profile --stats=true ./test