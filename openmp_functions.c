#include <stdio.h>
#include <omp.h>

void saludo(){
    printf("Hello from process: %d\n",omp_get_thread_num());
}
