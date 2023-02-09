# Programación Paralela 2022 - __TP Final__ _(OpenMP + CUDA)_
## _Consigna:_ Edge-detecting + Reverse edge-detecting
###  Partiendo del código de ejemplo en: https://github.com/steven-chien/DD2360-HT19.git, implementar los 3 pasos (grayscale, gauss filter y sobel filter) y agregar un 4to paso de  reverse edge-detecting

### Deben realizarse 4 implementaciones:
### __* CPU serie__
### __* CPU con openMP__
### __* CUDA con global memory__
### __* CUDA con shared memory__

### Analizar y documentar las diferencias de performance entre las 4 implementaciones

---
## _Archivos disponibles:_
- ### __Makefile__
- ### __./images/__
    - ### hk.bmp
    - ### hw3.bmp
    - ### nyc.bmp
    - ### rome.bmp
- ### __cpu_functions.cu__ _(Funciones sin paralelismo)_
- ### __gpu_functions.cu__ _(Funciones con paralelismo CUDA)_
- ### __openmp_functions.c__ _(Funciones con paralelismo OpenMP)_

- ## __hw3_NoCuda.cu__ _(Versión del proyecto que ejecuta los modulos CPU-Secuencial y OpenMP)_
- ## __hw3_ex1.cu__ _(Versión del proyecto que ejecuta los modulos CPU-Secuencial, OpenMP y Cuda)_
---
## _Observaciones y Conclusiones:_

...

---

- ## _Proyecto Original:_ __"Assigment III: Advanced Cuda" - KTH Royal Institute of Technology__
- ## _Modificaciones:_ Juan Miguel Badariotti - Estudiante de IComp en la __FCEFyN - UNC__


