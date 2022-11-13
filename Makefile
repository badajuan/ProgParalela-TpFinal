
compile:
	nvcc hw3_ex1.cu -o hw3 -Xcompiler -fopenmp

romeMP: clear mp
	@ ./mp ./images/rome.bmp $(T)

mp:
	@ nvcc hw3_NoCUDA.cu -o mp -Xcompiler -fopenmp

clear:
	@ rm ./mp -f

