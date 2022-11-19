# Nombre de la imagen a procesar
F=rome
# Flags de GCC
CFLAGS=-O3
# Cantidad de hilos a utilizar en OpenMP
T=16


# Para solo compilar la versión sólo OMP 		--> make omp
# Para compilar y correr la versión OMP			--> make OMP
# Para solo compilar la versión CUDA + OMP		--> make cuda
# Para compilar y correr la versión CUDA + OMP	--> make CUDA

# Para correr OMP sobre todas las imagenes 		  --> make all-OMP
# Para correr CUDA + OMP sobre todas las imagenes --> make all-CUDA

# Para eliminar todas las imagenes de resultados  --> make clean-r
# Para restaurar el proyecto al estado inicial	  --> make clean

default: cuda

CUDA: cuda
	@ ./hw3 ./images/$(F).bmp $(T)

OMP: omp
	@ ./omp ./images/$(F).bmp $(T)

all-CUDA: cuda
	@ ./hw3 ./images/hk.bmp $(T)
	@ ./hw3 ./images/hw3.bmp $(T)
	@ ./hw3 ./images/rome.bmp $(T)
	@ ./hw3 ./images/nyc.bmp $(T)

all-OMP: omp
	@ ./omp ./images/hk.bmp $(T)
	@ ./omp ./images/hw3.bmp $(T)
	@ ./omp ./images/rome.bmp $(T)
	@ ./omp ./images/nyc.bmp $(T)

omp: clean-omp
	@ echo "Optimización de GCC = '$(CFLAGS)'"
	@ nvcc hw3_NoCUDA.cu -o omp -Xcompiler -fopenmp $(CFLAGS)

cuda: clean-cuda
	@ echo "Optimización de GCC = '$(CFLAGS)'"
	@ nvcc hw3_ex1.cu -o hw3 -Xcompiler -fopenmp $(CFLAGS)

clean: clean-omp clean-cuda clean-r

clean-omp:
	@ rm ./omp -f

clean-cuda:
	@ rm ./hw3 -f

clean-r:
	@ rm ./images/*_cuda.bmp -f
	@ rm ./images/*_omp.bmp -f
	@ rm ./images/hw3_*.bmp -f


