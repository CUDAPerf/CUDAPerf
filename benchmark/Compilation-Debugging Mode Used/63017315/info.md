url
https://stackoverflow.com/questions/63017315/how-to-improve-computational-time-for-sorting-with-thrust
notes
nvcc -o withG -G test.cu
nvcc -o noG test.cu
nsys profile --stats=true ./withG  
nsys profile --stats=true ./noG
