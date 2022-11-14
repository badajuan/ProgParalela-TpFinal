
compile:
	nvcc hw3_ex1.cu -o hw3 -Xcompiler -fopenmp

allMP: clear romeMP nycMP

romeMP: mp
	@ ./mp ./images/rome.bmp $(T)

nycMP: mp
	@ ./mp ./images/nyc.bmp $(T)

mp:
	@ nvcc hw3_NoCUDA.cu -o mp -Xcompiler -fopenmp

clear:
	@ rm ./mp -f
	@ rm ./hw3 -f

