
default: compile

allMP: clear romeMP nycMP

all: clear rome nyc

rome: compile
	@ ./hw3 ./images/rome.bmp $(T)

nyc: compile
	@ ./hw3 ./images/nyc.bmp $(T)

romeMP: mp
	@ ./hw3 ./images/rome.bmp $(T)

nycMP: mp
	@ ./mp ./images/nyc.bmp $(T)

mp: clear
	@ nvcc hw3_NoCUDA.cu -o mp -Xcompiler -fopenmp

compile: clear
	@ nvcc hw3_ex1.cu -o hw3 -Xcompiler -fopenmp

clear:
	@ rm ./mp -f
	@ rm ./hw3 -f

