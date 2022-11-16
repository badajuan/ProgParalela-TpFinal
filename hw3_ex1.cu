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
#define bold        "\x1B[1m"     
#define normal      "\x1B[0m"

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
    for (int h = 0; h < alto; h++) {
        int offset = h * ancho;

        for (int w = 0; w < ancho; w++){
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
    
    if(cudaError != cudaSuccess) {
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
void guardar_resultado(int index, double tiempo_cpu, double tiempo_gpu_G, double tiempo_gpu_S,
                    double tiempo_openmp, int ancho, int alto, float *image_cuda, float *image_omp){
    char path1[255];
    char path2[255];
    //char path3[255];
    

    sprintf(path1, "images/%d_omp.bmp", index);
    writeBMPGrayscale(ancho, alto, image_omp, path1);
    sprintf(path2, "images/%d_cuda.bmp", index);
    writeBMPGrayscale(ancho, alto, image_cuda, path2);
    
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
    }
    printf("    Tiempo en CPU:          %fms\n",tiempo_cpu);
    
    printf("    Tiempo en OpenMP:       %fms\n",tiempo_openmp);
    printf("        Speedup: %.2f%%\n",(tiempo_cpu/tiempo_openmp -1)*100);

    if (tiempo_gpu_G == 0) {
        printf("    [Versión en GPU no disponible]\n");
    }
    else if(tiempo_gpu_G==tiempo_gpu_S){
        printf("    Tiempo en GPU:          %fms\n",tiempo_gpu_G);
        printf("        Speedup: %.2f%%\n",(tiempo_cpu/tiempo_gpu_G -1)*100);
        return;
    }
    else{
        printf("    Tiempo en GPU (Global): %fms\n",tiempo_gpu_G);
        printf("        Speedup: %.2f%%\n",(tiempo_cpu/tiempo_gpu_G -1)*100);
    }
    if(tiempo_gpu_S!=0){
        printf("    Tiempo en GPU (Shared): %fms\n",tiempo_gpu_S);
        printf("        Speedup: %.2f%%\n",(tiempo_cpu/tiempo_gpu_S -1)*100);
    }
    else{
        printf("    [Versión en GPU (Shared Memory) no disponible]\n");
    }
}

int main(int argc, char **argv){
    BMPImage bitmap          = { 0 };
    float    *d_bitmap       = { 0 };
    float    *image_cuda1[2]   = { 0 };
    //float    *image_cuda2[2]   = { 0 };
    float    *image_omp[2]   = { 0 };
    float    *d1_image_out[2] = { 0 };
    float    *d2_image_out[2] = { 0 };
    int      image_size      =   0;
    tval     t[2]            = { 0 };
    double   elapsed[4]      = { 0 };
    double   suma[4]         = { 0 };
    int      threads         =  16;
    dim3     grid(1);                       // The grid will be defined later
    dim3     block(BLOCK_SIZE, BLOCK_SIZE); // The block size will not change
    
    // Make sure the filename is provided
    if (argc == 1){
        fprintf(stderr, "Error: The filename is missing!\n");
        return -1;
    }
    else if(argc==3){ //Si me pasan la cantidad de threads a utilizar
        threads=atoi(argv[2]);
    }
    if(access(argv[1],F_OK)==-1){ // Chequeo si la imagen a abrir existe
        printf("%sPath '%s' inválido%s - Intente nuevamente\n",bold,argv[1],normal);
        return -1;
    }
    
    // Read the input image and update the grid dimension
    bitmap     = readBMP(argv[1]);
    image_size = bitmap.ancho * bitmap.alto;
    grid       = dim3(((bitmap.ancho  + (BLOCK_SIZE - 1)) / BLOCK_SIZE),
                      ((bitmap.alto   + (BLOCK_SIZE - 1)) / BLOCK_SIZE));

    printf("Imagen '%s' abierta (Ancho = %dp - Alto = %dp) | ",argv[1]+9,bitmap.ancho, bitmap.alto);
    printf("Número de hilos: %d\n",threads);

    // Allocate the intermediate image buffers for each step
    for (int i = 0; i < 2; i++){
        image_cuda1[i] = (float *)calloc(image_size, sizeof(float));
        //image_cuda2[i] = (float *)calloc(image_size, sizeof(float));
        image_omp[i] = (float *)calloc(image_size, sizeof(float));
        cudaMalloc(&d1_image_out[i], image_size * sizeof(float));
        cudaMemset(d1_image_out[i], 0, image_size * sizeof(float));
        cudaMalloc(&d2_image_out[i], image_size * sizeof(float));
        cudaMemset(d2_image_out[i], 0, image_size * sizeof(float));
    }

    cudaMalloc(&d_bitmap, image_size * sizeof(float) * 3);
    cudaMemcpy(d_bitmap, bitmap.data,
               image_size * sizeof(float) * 3, cudaMemcpyHostToDevice);
    
    // Step 1: Convert to grayscale
    {
        //Launch the CPU version
        gettimeofday(&t[0], NULL);
        cpu_grayscale(bitmap.ancho, bitmap.alto, bitmap.data, image_omp[0]);
        gettimeofday(&t[1], NULL);        
        elapsed[0] = get_elapsed(t[0], t[1]);

        //Launch the OpenMP version
        gettimeofday(&t[0], NULL);
        openmp_grayscale(bitmap.ancho, bitmap.alto, bitmap.data, image_omp[0],threads);
        gettimeofday(&t[1], NULL);
        elapsed[3] = get_elapsed(t[0], t[1]);

        // Launch the GPU version
        gettimeofday(&t[0], NULL);
        gpu_grayscale<<<grid, block>>>(bitmap.ancho, bitmap.alto,d_bitmap, d1_image_out[0]);
        cudaMemcpy(image_cuda1[0], d1_image_out[0],image_size * sizeof(float), cudaMemcpyDeviceToHost);
        gettimeofday(&t[1], NULL);
        elapsed[1] = get_elapsed(t[0], t[1]);
        elapsed[2] = elapsed[1];
        
        //Hago el mismo procedimiento para la imagen que usa SM
        gpu_grayscale<<<grid, block>>>(bitmap.ancho, bitmap.alto,d_bitmap, d2_image_out[0]); 
        //cudaMemcpy(image_cuda2[0], d1_image_out[0],image_size * sizeof(float), cudaMemcpyDeviceToHost);
        
        // Store the result image in grayscale
        guardar_resultado(1, elapsed[0], elapsed[1], elapsed[2], elapsed[3], bitmap.ancho, bitmap.alto, image_cuda1[0],image_omp[0]);
        suma[0]+=elapsed[0];suma[1]+=elapsed[1];suma[2]+=elapsed[2];suma[3]+=elapsed[3];   
    }

    // Step 2: Apply a 3x3 Gaussian filter
    {
        // Launch the CPU version
        gettimeofday(&t[0], NULL);
        cpu_gaussian(bitmap.ancho, bitmap.alto, image_omp[0], image_omp[1]);
        gettimeofday(&t[1], NULL);
        elapsed[0] = get_elapsed(t[0], t[1]);
        
        //Launch the OpenMP version
        gettimeofday(&t[0], NULL);
        openmp_gaussian(bitmap.ancho, bitmap.alto, image_omp[0], image_omp[1],threads);
        gettimeofday(&t[1], NULL);
        elapsed[3] = get_elapsed(t[0], t[1]);

        // Launch the GPU-GM version
        gettimeofday(&t[0], NULL);
        gpu_gaussian_GM<<<grid, block>>>(bitmap.ancho, bitmap.alto,d1_image_out[0], d1_image_out[1]);
        cudaMemcpy(image_cuda1[1], d1_image_out[1],image_size * sizeof(float), cudaMemcpyDeviceToHost);
        gettimeofday(&t[1], NULL);
        elapsed[1] = get_elapsed(t[0], t[1]);

        // Launch the GPU-SM version
        gettimeofday(&t[0], NULL);
        gpu_gaussian_SM<<<grid, block>>>(bitmap.ancho, bitmap.alto,d2_image_out[0], d2_image_out[1]);
        cudaMemcpy(image_cuda1[1], d2_image_out[1],image_size * sizeof(float), cudaMemcpyDeviceToHost);
        gettimeofday(&t[1], NULL);
        elapsed[2] = get_elapsed(t[0], t[1]);
        
        
        // Store the result image with the Gaussian filter applied
        guardar_resultado(2, elapsed[0], elapsed[1], elapsed[2], elapsed[3], bitmap.ancho, bitmap.alto, image_cuda1[1],image_omp[1]);
        suma[0]+=elapsed[0];suma[1]+=elapsed[1];suma[2]+=elapsed[2];suma[3]+=elapsed[3];   
    }
    

    // Step 3: Apply a Sobel filter
    {
        // Launch the CPU version
        gettimeofday(&t[0], NULL);
        cpu_sobel(bitmap.ancho, bitmap.alto, image_omp[1], image_omp[0]);
        gettimeofday(&t[1], NULL);
        elapsed[0] = get_elapsed(t[0], t[1]);
        
        //Launch the OpenMP version
        gettimeofday(&t[0], NULL);
        openmp_sobel(bitmap.ancho, bitmap.alto, image_omp[1],image_omp[0],threads);
        gettimeofday(&t[1], NULL);
        elapsed[3] = get_elapsed(t[0], t[1]);

        // Launch the GPU-GM version
        gettimeofday(&t[0], NULL);
        gpu_sobel_GM<<<grid, block>>>(bitmap.ancho, bitmap.alto,d1_image_out[1], d1_image_out[0]);
        cudaMemcpy(image_cuda1[0], d1_image_out[0],image_size * sizeof(float), cudaMemcpyDeviceToHost);
        gettimeofday(&t[1], NULL);
        elapsed[1] = get_elapsed(t[0], t[1]);
        
        // Launch the GPU-SM version
        gettimeofday(&t[0], NULL);
        gpu_sobel_SM<<<grid, block>>>(bitmap.ancho, bitmap.alto,d2_image_out[1], d2_image_out[0]);
        cudaMemcpy(image_cuda1[0], d2_image_out[0],image_size * sizeof(float), cudaMemcpyDeviceToHost);
        gettimeofday(&t[1], NULL);
        elapsed[2] = get_elapsed(t[0], t[1]);

        // Store the final result image with the Sobel filter applied
        guardar_resultado(3, elapsed[0], elapsed[1], elapsed[2], elapsed[3], bitmap.ancho, bitmap.alto, image_cuda1[0],image_omp[0]);
        suma[0]+=elapsed[0];suma[1]+=elapsed[1];suma[2]+=elapsed[2];suma[3]+=elapsed[3];
    }
    
    printf("\nTiempo total en ejecución secuencial:             %.3fms\n",suma[0]);
    printf("\nTiempo total usando paralelismo de OpenMP:        %.3fms\n",suma[3]);
    printf("    (Speedup total de %.2f%%)\n",(suma[0]/suma[3] -1)*100);
    printf("\nTiempo total usando paralelismo de CUDA (Global): %.3fms\n",suma[1]);
    printf("    (Speedup total de %2.5f%%)\n",(suma[0]/suma[1] -1)*100);
    //printf("\n[Versión en GPU (Shared Memory) no disponible]\n");
    printf("\nTiempo total usando paralelismo de CUDA (Shared): %.3fms\n",suma[2]);
    printf("    (Speedup total de %2.5f%%)\n",(suma[0]/suma[2] -1)*100);

    // Release the allocated memory
    for (int i = 0; i < 2; i++){
        free(image_cuda1[i]);
        //free(image_cuda2[i]);
        free(image_omp[i]);
        cudaFree(d1_image_out[i]);
        cudaFree(d2_image_out[i]);
    }
    
    freeBMP(bitmap);
    cudaFree(d_bitmap);
    printf("\n");
    return 0;
}

