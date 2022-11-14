F=rome

default: compile omp

CUDA: hw3
	@ ./hw3 ./images/$(F).bmp $(T)

OMP: omp
	@ ./omp ./images/$(F).bmp $(T)

all: clear-hw3 rome nyc
	@ ./hw3 ./images/hk.bmp $(T)
	@ ./hw3 ./images/hw3.bmp $(T)

rome: compile
	@ ./hw3 ./images/rome.bmp $(T)

nyc: compile
	@ ./hw3 ./images/nyc.bmp $(T)

allOOMP: clear-omp romeOMP nycOMP
	@ ./omp ./images/hk.bmp $(T)
	@ ./omp ./images/hw3.bmp $(T)

romeOMP: omp
	@ ./omp ./images/rome.bmp $(T)

nycOMP: omp
	@ ./omp ./images/nyc.bmp $(T)

omp: clear-omp
	@ nvcc hw3_NoCUDA.cu -o omp -Xcompiler -fopenmp

compile: clear-hw3
	@ nvcc hw3_ex1.cu -o hw3 -Xcompiler -fopenmp

clear: clear-omp clear-hw3

clear-omp:
	@ rm ./omp -f

clear-hw3:
	@ rm ./hw3 -f

