#define BLOCK_SIZE_SH 18


/*
* Probé a ver si usar memoria constante para los arreglos mejoraba los tiempos pero no hay mejora significativa
__constant__ float gaussian[9] = { 1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f,
                                   2.0f / 16.0f, 4.0f / 16.0f, 2.0f / 16.0f,
                                   1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f };

__constant__ float sobel_x[9] = { 1.0f,  0.0f, -1.0f,
                                  2.0f,  0.0f, -2.0f,
                                  1.0f,  0.0f, -1.0f };

__constant__ float sobel_y[9] = { 1.0f,  2.0f,  1.0f,
                                  0.0f,  0.0f,  0.0f,
                                 -1.0f, -2.0f, -1.0f };                         
*/ 

/**
 * Converts a given 24bpp image into 8bpp grayscale using the GPU.
 */
__global__ void gpu_grayscale(int ancho, int alto, float *image, float *image_out){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if( x < ancho && y < alto){
        float *pixel = &image[(y*ancho + x)*3]; //Multiplico por 3 para tener en cuenta la linealización del arreglo
        image_out[(y*ancho + x)]=   pixel[0] * 0.0722f + // B
                                    pixel[1] * 0.7152f + // G
                                    pixel[2] * 0.2126f;  // R
    }
}

/**
 * Applies a 3x3 convolution matrix to a pixel using the GPU. 
 * Cada thread llama a este procedimiento por pixel, así que esta convolución se serializa para cada pixel pero se paraleliza previamente por los metodos anteriores.
 */
__device__ float gpu_applyFilter(float *image, int stride, float *matrix, int filter_dim){  
    
    float pixel = 0.0f;
   
    for (int h = 0; h < filter_dim; h++){
        int offset        = h * stride;
        int offset_kernel = h * filter_dim;
        
        for (int w = 0; w < filter_dim; w++){
            pixel += image[offset + w] * matrix[offset_kernel + w];
        }
    }
    return pixel;
}

/**
 * Applies a Gaussian 3x3 filter to a given image using the GPU. Versión con implementación de SHARED MEMORY
 */
__global__ void gpu_gaussian_SM(int ancho, int alto, float *image, float *image_out){
    float gaussian[9] = { 1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f,
                                   2.0f / 16.0f, 4.0f / 16.0f, 2.0f / 16.0f,
                                   1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f };

    __shared__ float sh_block[BLOCK_SIZE_SH * BLOCK_SIZE_SH];                      
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;


    if (x < (ancho - 2) && y < (alto - 2)) {
        int offset_t = y * ancho + x;
        int offset_s = threadIdx.y * BLOCK_SIZE_SH + threadIdx.x;
        int offset   = (y + 1) * ancho + (x + 1);
        
        sh_block[threadIdx.y*BLOCK_SIZE_SH + threadIdx.x] = image[offset_t]; //Cada bloque de shared memory se matchea con un bloque y/x y cada thread copia su pixel asignado al bloque de SM. Como la shared es compartida solo entre bloques de threads no se "pisan" entre distintos bloques.
        __syncthreads(); //Para asegurarme que todos los threads del bloque copiaron su pixel al arreglo de la SharedMemory

        
        image_out[offset] = gpu_applyFilter(&sh_block[offset_s], BLOCK_SIZE_SH, gaussian, 3);
    }
}

/**
 * Applies a Gaussian 3x3 filter to a given image using the GPU. Versión con implementación de GLOBAL MEMORY
 */
__global__ void gpu_gaussian_GM(int ancho, int alto, float *image, float *image_out){
    float gaussian[9] = { 1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f,
                                   2.0f / 16.0f, 4.0f / 16.0f, 2.0f / 16.0f,
                                   1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f };
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < (ancho - 2) && y < (alto - 2)) 
    {
        int offset_t = y * ancho + x;
        int offset   = (y + 1) * ancho + (x + 1);
        
        image_out[offset] = gpu_applyFilter(&image[offset_t],
                                            ancho, gaussian, 3);
    }
}

/**
 * Calculates the gradient of an image using a Sobel filter on the GPU. Versión con implementación de SHARED MEMORY
 */
__global__ void gpu_sobel_SM(int ancho, int alto, float *image, float *image_out){
    float sobel_x[9] = { 1.0f,  0.0f, -1.0f,
                         2.0f,  0.0f, -2.0f,
                         1.0f,  0.0f, -1.0f };
    float sobel_y[9] = { 1.0f,  2.0f,  1.0f,
                         0.0f,  0.0f,  0.0f,
                        -1.0f, -2.0f, -1.0f };

    __shared__ float sh_block[BLOCK_SIZE_SH * BLOCK_SIZE_SH];
    
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(y<alto-2 && x< ancho-2){
        int offset_t = y * ancho + x;
        int offset_s = threadIdx.y * BLOCK_SIZE_SH + threadIdx.x;
        int offset   = (y + 1) * ancho + x + 1;

        sh_block[threadIdx.y*BLOCK_SIZE_SH + threadIdx.x] = image[offset_t]; //Cada bloque de shared memory se matchea con un bloque y/x y cada thread copia su pixel asignado al bloque de SM. Como la shared es compartida solo entre bloques de threads no se "pisan" entre distintos bloques.
        __syncthreads(); //Para asegurarme que todos los threads del bloque copiaron su pixel al arreglo de la SharedMemory
        
        float gx = gpu_applyFilter(&sh_block[offset_s], BLOCK_SIZE_SH, sobel_x, 3);
        float gy = gpu_applyFilter(&sh_block[offset_s], BLOCK_SIZE_SH, sobel_y, 3);
            
            // Note: The output can be negative or exceed the max. color value
            // of 255. We compensate this afterwards while storing the file.
        image_out[offset] = sqrtf(gx * gx + gy * gy);
        
    }
}

/**
 * Calculates the gradient of an image using a Sobel filter on the GPU. Versión con implementación de GLOBAL MEMORY
 */
__global__ void gpu_sobel_GM(int ancho, int alto, float *image, float *image_out){
    float sobel_x[9] = { 1.0f,  0.0f, -1.0f,
                         2.0f,  0.0f, -2.0f,
                         1.0f,  0.0f, -1.0f };
    float sobel_y[9] = { 1.0f,  2.0f,  1.0f,
                         0.0f,  0.0f,  0.0f,
                        -1.0f, -2.0f, -1.0f };
    
    int h= blockIdx.y * blockDim.y + threadIdx.y;
    int w = blockIdx.x * blockDim.x + threadIdx.x;

    if(h<alto-2 && w< ancho-2){

        int offset_t = h * ancho;
        int offset   = (h + 1) * ancho;
        
        float gx = gpu_applyFilter(&image[offset_t + w], ancho, sobel_x, 3);
        float gy = gpu_applyFilter(&image[offset_t + w], ancho, sobel_y, 3);
            
            // Note: The output can be negative or exceed the max. color value
            // of 255. We compensate this afterwards while storing the file.
        image_out[offset + (w + 1)] = sqrtf(gx * gx + gy * gy);
        
    }
}