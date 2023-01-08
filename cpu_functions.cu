
/**
 * Converts a given 24bpp image into 8bpp grayscale using the CPU.
 */
void cpu_grayscale(int ancho, int alto, float *image, float *image_out){
    for (int y = 0; y < alto; y++)
    {
        int offset_out = y * ancho;      // 1 color per pixel
        int offset     = offset_out * 3; // 3 colors per pixel
        
        for (int x = 0; x < ancho; x++)
        {
            float *pixel = &image[offset + x * 3];
            
            // Convert to grayscale following the "luminance" model
            image_out[offset_out + x] = pixel[0] * 0.0722f + // B
                                        pixel[1] * 0.7152f + // G
                                        pixel[2] * 0.2126f;  // R
        }
    }
}

/**
 * Applies a 3x3 convolution matrix to a pixel using the CPU.
 */
float cpu_applyFilter(float *image, int stride, float *matrix, int filter_dim){
    float pixel = 0.0f;
    
    for (int h = 0; h < filter_dim; h++)
    {
        int offset        = h * stride;
        int offset_kernel = h * filter_dim;
        
        for (int w = 0; w < filter_dim; w++)
        {
            pixel += image[offset + w] * matrix[offset_kernel + w];
        }
    }
    
    return pixel;
}

/**
 * Applies a Gaussian 3x3 filter to a given image using the CPU.
 */
void cpu_gaussian(int ancho, int alto, float *image, float *image_out){
    float gaussian[9] = { 1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f,
                          2.0f / 16.0f, 4.0f / 16.0f, 2.0f / 16.0f,
                          1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f };
    
    for (int h = 0; h < (alto - 2); h++)
    {
        int offset_t = h * ancho;
        int offset   = (h + 1) * ancho;
        
        for (int w = 0; w < (ancho - 2); w++)
        {
            image_out[offset + (w + 1)] = cpu_applyFilter(&image[offset_t + w],
                                                          ancho, gaussian, 3);
        }
    }
}

/**
 * Calculates the gradient of an image using a Sobel filter on the CPU.
 */
void cpu_sobel(int ancho, int alto, float *image, float *image_out){
    float sobel_x[9] = { 1.0f,  0.0f, -1.0f,
                         2.0f,  0.0f, -2.0f,
                         1.0f,  0.0f, -1.0f };
    float sobel_y[9] = { 1.0f,  2.0f,  1.0f,
                         0.0f,  0.0f,  0.0f,
                        -1.0f, -2.0f, -1.0f };
    
    for (int h = 0; h < (alto - 2); h++)
    {
        int offset_t = h * ancho;
        int offset   = (h + 1) * ancho;
        
        for (int w = 0; w < (ancho - 2); w++)
        {
            float gx = cpu_applyFilter(&image[offset_t + w], ancho, sobel_x, 3);
            float gy = cpu_applyFilter(&image[offset_t + w], ancho, sobel_y, 3);
            
            // Note: The output can be negative or exceed the max. color value
            // of 255. We compensate this afterwards while storing the file.
            image_out[offset + (w + 1)] = sqrtf(gx * gx + gy * gy);
        }
    }
}

void cpu_alternative_sobel(int ancho, int alto, float *image, float *image_out){
    //Algoritmo de referencia:
    //edgei;j = imagei-1;j + imagei+1;j + imagei;j-1 + imagei;j+1 -4imagei;j
    for (int y = 1; y < (alto - 1); y++){
        for (int x = 1; x < (ancho - 1); x++){
            int offset = y * ancho + x;
            int partial=image[(y-1)*ancho+x]+image[(y+1)*ancho+x]+image[offset+1]+image[offset-1]-4*image[offset];
            if(partial<3){
                image_out[offset]=partial+10;
            }
            else{
                image_out[offset]=partial+100;
            }
        }
    }
}


void cpu_inverse_sobel(int ancho, int alto, float *image, float *image_out){
    //Algoritmo de referencia:
    // imagei;j = (imagei-1;j + imagei+1;j + imagei;j-1 + imagei;j+1 - edgei;j)/4
    for (int y = 0; y < (alto - 1); y++){
        for (int x = 0; x < (ancho - 1); x++){
            int offset = y * ancho + x;
            int partial=(image[(y-1)*ancho+x]+image[(y+1)*ancho+x]+image[offset-1]+image[offset+1]-image[offset]);
            //if(partial<10){
                image_out[offset]=partial;
            //}
            /*
            else if(partial<100){
                image_out[offset]=partial+25;
            else{
                image_out[offset]=partial+75;
            }
            //image_out[offset]=255-image[offset];
            */            
        }
    }
}
