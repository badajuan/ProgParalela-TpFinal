#include <stdio.h>
#include <omp.h>

/**
 * Converts a given 24bpp image into 8bpp grayscale using the openmp.
 */
void openmp_grayscale(int ancho, int alto, float *image, float *image_out,int threads){
    #pragma omp parallel for num_threads(threads)    
    for (int y = 0; y < alto; y++){
        int offset_out = y * ancho;      // 1 color per pixel
        int offset     = offset_out * 3; // 3 colors per pixel
        
        for (int x = 0; x < ancho; x++) {
            float *pixel = &image[offset + x * 3];
            
            // Convert to grayscale following the "luminance" model
            image_out[offset_out + x] = pixel[0] * 0.0722f + // B
                                        pixel[1] * 0.7152f + // G
                                        pixel[2] * 0.2126f;  // R
        }
    }
}

/**
 * Applies a 3x3 convolution matrix to a pixel using the openmp.
 */
float openmp_applyFilter(float *image, int stride, float *matrix, int filter_dim){
    float pixel = 0.0f;
    //#pragma omp parallel for num_threads(2)
    for (int h = 0; h < filter_dim; h++){
        int offset        = h * stride;
        int offset_kernel = h * filter_dim;

        for (int w = 0; w < filter_dim; w++){
            //#pragma omp atomic
            pixel += image[offset + w] * matrix[offset_kernel + w];
        }

    }
    return pixel;
}

/**
 * Applies a Gaussian 3x3 filter to a given image using the openmp.
 */
void openmp_gaussian(int ancho, int alto, float *image, float *image_out, int threads){
    float gaussian[9] = { 1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f,
                          2.0f / 16.0f, 4.0f / 16.0f, 2.0f / 16.0f,
                          1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f };

    #pragma omp parallel for num_threads(threads)
    for (int h = 0; h < (alto - 2); h++){
        int offset_t = h * ancho;
        int offset   = (h + 1) * ancho;
        
        for (int w = 0; w < (ancho - 2); w++)
        {
            image_out[offset + (w + 1)] = openmp_applyFilter(&image[offset_t + w],
                                                          ancho, gaussian, 3);
        }
    }
}

/**
 * Calculates the gradient of an image using a Sobel filter on the openmp.
 */
void openmp_sobel(int ancho, int alto, float *image, float *image_out, int threads){
    float sobel_x[9] = { 1.0f,  0.0f, -1.0f,
                         2.0f,  0.0f, -2.0f,
                         1.0f,  0.0f, -1.0f };
    float sobel_y[9] = { 1.0f,  2.0f,  1.0f,
                         0.0f,  0.0f,  0.0f,
                        -1.0f, -2.0f, -1.0f };

    #pragma omp parallel for num_threads(threads)
    for (int h = 0; h < (alto - 2); h++) {
        int offset_t = h * ancho;
        int offset   = (h + 1) * ancho;

        for (int w = 0; w < (ancho - 2); w++) {
            float gx = openmp_applyFilter(&image[offset_t + w], ancho, sobel_x, 3);
            float gy = openmp_applyFilter(&image[offset_t + w], ancho, sobel_y, 3);
            
            // Note: The output can be negative or exceed the max. color value
            // of 255. We compensate this afterwards while storing the file.
            image_out[offset + (w + 1)] = sqrtf(gx * gx + gy * gy);
        }
    }
}