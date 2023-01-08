#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>

#include "openmp_functions.c"
#include "cpu_functions.cu"
#include "gpu_functions.cu"

#define BLOCK_SIZE  16
#define HEADER_SIZE 138

typedef unsigned char BYTE;

/**
 * Structure that represents a BMP image.
 */
typedef struct {
    int   ancho;
    int   alto;
    float *data;
} BMPImage;

typedef struct timeval tval;

BYTE g_info[HEADER_SIZE]; // Reference header

/**
 * Reads a BMP 24bpp file and returns a BMPImage structure.
 * Thanks to https://stackoverflow.com/a/9296467
 */
BMPImage readBMP(char *filename){
    BMPImage bitmap = { 0 };
    int      size   = 0;
    BYTE     *data  = NULL;
    FILE     *file  = fopen(filename, "rb");
    
    // Read the header (expected BGR - 24bpp)
    if(!fread(g_info, sizeof(BYTE), HEADER_SIZE, file)){
        exit(1);
    }

    // Get the image ancho / alto from the header
    bitmap.ancho  = *((int *)&g_info[18]);
    bitmap.alto = *((int *)&g_info[22]);
    size          = *((int *)&g_info[34]);
    
    // Read the image data
    data = (BYTE *)malloc(sizeof(BYTE) * size);
    if(!fread(data, sizeof(BYTE), size, file)){
        exit(1);
    }
    
    // Convert the pixel values to float
    bitmap.data = (float *)malloc(sizeof(float) * size);
    
    for (int i = 0; i < size; i++)
    {
        bitmap.data[i] = (float)data[i];
    }
    
    fclose(file);
    free(data);
    
    return bitmap;
}

/**
 * Writes a BMP file in grayscale given its image data and a filename.
 */
void writeBMPGrayscale(int ancho, int alto, float *image, char *filename){
    FILE *file = NULL;
    
    file = fopen(filename, "wb");
    
    // Write the reference header
    fwrite(g_info, sizeof(BYTE), HEADER_SIZE, file);
    
    // Unwrap the 8-bit grayscale into a 24bpp (for simplicity)
    for (int h = 0; h < alto; h++)
    {
        int offset = h * ancho;
        
        for (int w = 0; w < ancho; w++)
        {
            BYTE pixel = (BYTE)((image[offset + w] > 255.0f) ? 255.0f :
                                (image[offset + w] < 0.0f)   ? 0.0f   :
                                                               image[offset + w]);
            
            // Repeat the same pixel value for BGR
            fputc(pixel, file);
            fputc(pixel, file);
            fputc(pixel, file);
        }
    }
    
    fclose(file);
}

/**
 * Releases a given BMPImage.
 */
void freeBMP(BMPImage bitmap){
    free(bitmap.data);
}

/**
 * Checks if there has been any CUDA error. The method will automatically print
 * some information and exit the program when an error is found.
 */
void checkCUDAError(){
    cudaError_t cudaError = cudaGetLastError();
    
    if(cudaError != cudaSuccess)
    {
        printf("CUDA Error: Returned %d: %s\n", cudaError,
                                                cudaGetErrorString(cudaError));
        exit(-1);
    }
}

/**
 * Calculates the elapsed time between two time intervals (in milliseconds).
 */
double get_elapsed(tval t0, tval t1){
    return (double)(t1.tv_sec - t0.tv_sec) * 1000.0L + (double)(t1.tv_usec - t0.tv_usec) / 1000.0L;
}

/**
 * Stores the result image and prints a message.
 */
void store_result(int index, double elapsed_cpu,
                    double elapsed_openmp, int ancho, int alto, float *image){
    char path[255];
    
    sprintf(path, "images/%d_omp.bmp", index);
    writeBMPGrayscale(ancho, alto, image, path);
    
    //printf("Step #%d Completed - Result stored in \"%s\".\n", index, path);
    printf("\nPaso #%d Completado:",index);
    switch(index){
        case 1:
            printf(" Escala de Grises\n");
            break;
        case 2:
            printf(" Blur de Gauss\n");
            break;
        case 3:
            printf(" Filtro de Sobel\n");
            break;
        case 4:
            printf(" Filtro de Sobel Inverso\n");
            break;
    }
    printf("    Tiempo en CPU:      %fms\n",elapsed_cpu);
    
    printf("    Tiempo en OpenMP:   %fms\n",elapsed_openmp);
    printf("        Speedup: %.2f%%\n",(elapsed_cpu/elapsed_openmp -1)*100);
}

int main(int argc, char **argv){
    BMPImage bitmap          = { 0 };
    float    *image_out[2]   = { 0 };
    int      image_size      = 0;
    tval     t[2]            = { 0 };
    double   elapsed[2]      = { 0 };
    double   suma[2]         = {0,0};
    int      threads         = 16;
    
    // Make sure the filename is provided
    if (argc == 1){
        fprintf(stderr, "Error: The filename is missing!\n");
        return -1;
    }
    else if(argv[2]!=NULL){
        threads=atoi(argv[2]);
    }
    if(access(argv[1],F_OK)==-1){
        printf("    Path '%s' inválido - Intente nuevamente\n",argv[1]);
        return 1;
    }
    
    // Read the input image and update the grid dimension
    bitmap     = readBMP(argv[1]);
    image_size = bitmap.ancho * bitmap.alto;
    printf("Imagen '%s' abierta (Ancho = %dp - Alto = %dp) | ",argv[1]+9,bitmap.ancho, bitmap.alto);
    printf("Número de hilos: %d\n",threads);
    
    // Allocate the intermediate image buffers for each step
    for (int i = 0; i < 2; i++){
        image_out[i] = (float *)calloc(image_size, sizeof(float));
    }

    // Step 1: Convert to grayscale
    {
        //Launch the CPU version
        gettimeofday(&t[0], NULL);
        cpu_grayscale(bitmap.ancho, bitmap.alto, bitmap.data, image_out[0]);
        gettimeofday(&t[1], NULL);
        
        elapsed[0] = get_elapsed(t[0], t[1]);

        //Launch the OpenMP version
        gettimeofday(&t[0], NULL);
        openmp_grayscale(bitmap.ancho, bitmap.alto, bitmap.data, image_out[0],threads);
        gettimeofday(&t[1], NULL);
        
        elapsed[1] = get_elapsed(t[0], t[1]);

        // Store the result image in grayscale
        store_result(1, elapsed[0], elapsed[1], bitmap.ancho, bitmap.alto, image_out[0]);
        suma[0]+=elapsed[0];suma[1]+=elapsed[1];
    }

    
    // Step 2: Apply a 3x3 Gaussian filter
    {
        // Launch the CPU version
        gettimeofday(&t[0], NULL);
        cpu_gaussian(bitmap.ancho, bitmap.alto, image_out[0], image_out[1]);
        gettimeofday(&t[1], NULL);
        
        elapsed[0] = get_elapsed(t[0], t[1]);

        //Launch the OpenMP version
        gettimeofday(&t[0], NULL);
        openmp_gaussian(bitmap.ancho, bitmap.alto, image_out[0], image_out[1],threads);
        gettimeofday(&t[1], NULL);
        
        elapsed[1] = get_elapsed(t[0], t[1]);
        
        // Store the result image with the Gaussian filter applied
        store_result(2, elapsed[0], elapsed[1], bitmap.ancho, bitmap.alto, image_out[1]);
        suma[0]+=elapsed[0];suma[1]+=elapsed[1];
    }
    
    
    // Step 3: Apply a Sobel filter
    {
        // Launch the CPU version
        gettimeofday(&t[0], NULL);
        cpu_sobel(bitmap.ancho, bitmap.alto, image_out[1], image_out[0]);
        gettimeofday(&t[1], NULL);
        
        elapsed[0] = get_elapsed(t[0], t[1]);

        //Launch the OpenMP version
        gettimeofday(&t[0], NULL);
        openmp_sobel(bitmap.ancho, bitmap.alto, image_out[1],image_out[0],threads);
        gettimeofday(&t[1], NULL);
        
        elapsed[1] = get_elapsed(t[0], t[1]);
        
        // Store the final result image with the Sobel filter applied
        store_result(3, elapsed[0], elapsed[1], bitmap.ancho, bitmap.alto, image_out[0]);
        suma[0]+=elapsed[0];suma[1]+=elapsed[1];
    }
    

    // Step 4: Apply an Inverse-Sobel filter
    {
        // Launch the CPU version
        gettimeofday(&t[0], NULL);        
        //cpu_sobel(bitmap.ancho, bitmap.alto, image_out[1], image_out[0]);
        //cpu_alternative_sobel(bitmap.ancho, bitmap.alto, image_out[1], image_out[0]);
        cpu_inverse_sobel(bitmap.ancho, bitmap.alto, image_out[0], image_out[1]);
        //cpu_inverse_sobel(bitmap.ancho, bitmap.alto, image_out[0], image_out[1]);
        gettimeofday(&t[1], NULL);

        elapsed[0] = get_elapsed(t[0], t[1]);

        store_result(4, elapsed[0], elapsed[1], bitmap.ancho, bitmap.alto, image_out[1]);

    }

    printf("\nTiempo total en ejecución secuencial:       %.3fms\n",suma[0]);
    printf("\nTiempo total usando paralelismo de OpenMP:  %.3fms\n",suma[1]);
    printf("    Speedup total de %.2f%%\n",(suma[0]/suma[1] -1)*100);
    
    // Release the allocated memory
    for (int i = 0; i < 2; i++){
        free(image_out[i]);
    }
    
    freeBMP(bitmap);
    printf("\n");
    return 0;
}

