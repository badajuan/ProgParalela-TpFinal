F=rome

default: compile omp

CUDA:
	@ ./hw3 ./images/$(F).bmp $(T)

OMP:
	@ ./omp ./images/$(F).bmp $(T)

all: clear-cuda rome nyc
	@ ./hw3 ./images/hk.bmp $(T)
	@ ./hw3 ./images/hw3.bmp $(T)

rome: cuda
	@ ./hw3 ./images/rome.bmp $(T)

nyc: cuda
	@ ./hw3 ./images/nyc.bmp $(T)

allOMP: clear-omp romeOMP nycOMP
	@ ./omp ./images/hk.bmp $(T)
	@ ./omp ./images/hw3.bmp $(T)

romeOMP: omp
	@ ./omp ./images/rome.bmp $(T)

nycOMP: omp
	@ ./omp ./images/nyc.bmp $(T)

omp: clear-omp
	@ nvcc hw3_NoCUDA.cu -o omp -Xcompiler -fopenmp $(CFLAGS)

cuda: clear-cuda
	@ nvcc hw3_ex1.cu -o hw3 -Xcompiler -fopenmp

clear: clear-omp clear-cuda clear-r

clear-omp:
	@ rm ./omp -f

clear-cuda:
	@ rm ./hw3 -f

clear-r:
	@ rm ./images/*_cuda.bmp -f
	@ rm ./images/*_omp.bmp -f


