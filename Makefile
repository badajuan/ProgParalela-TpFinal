default: compile

all: clear rome nyc

allMP: clear romeMP nycMP

rome: compile
	@ ./hw3 ./images/rome.bmp $(T)

nyc: compile
	@ ./hw3 ./images/nyc.bmp $(T)

romeMP: mp
	@ ./mp ./images/rome.bmp $(T)

nycMP: mp
	@ ./mp ./images/nyc.bmp $(T)

mp: clear
	@ nvcc hw3_NoCUDA.cu -o mp -Xcompiler -fopenmp

compile: clear
	@ nvcc hw3_ex1.cu -o hw3 -Xcompiler -fopenmp

clear:
	@ rm ./mp -f
	@ rm ./hw3 -f

