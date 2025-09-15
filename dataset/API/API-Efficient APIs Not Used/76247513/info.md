url
https://stackoverflow.com/questions/76247513/how-can-i-do-segmented-reduction-using-cuda-thrust
notes
nvcc -o test test.cu
./test
N and K respectively represent the following meanings: N： The number of groups representing data. The dataset is divided into N groups, each containing K elements. K： Indicate the number of elements in each group. That is to say, each set of data contains K elements. The main purpose of the program is to calculate the minimum value in each set of data.