#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE  16
#define GROUP_SIZE 2

__device__ int mandel(float c_re, float c_im, int count)
{
  float z_re = c_re, z_im = c_im; //re: real part,  im: imaginary part
  int i;
  for (i = 0; i < count; ++i)
  {

    if (z_re * z_re + z_im * z_im > 4.f)
      break;

    float new_re = z_re * z_re - z_im * z_im;
    float new_im = 2.f * z_re * z_im;
    z_re = c_re + new_re;
    z_im = c_im + new_im;
  }

  return i;
}

__global__ void mandelKernel(float lowerX, float lowerY, float stepX, float stepY, int resX, int resY, int maxIterations, int *device_result, size_t pitch) {
    // To avoid error caused by the floating number, use the following pseudo code
    //
    // float x = lowerX + thisX * stepX;
    // float y = lowerY + thisY * stepY;

    //thread_position_x: the position on the 2D thread hierarchy
    int thread_position_x = (blockIdx.x * blockDim.x + threadIdx.x) * GROUP_SIZE ; //threadIdx is the threadID in this specific block
    int thread_position_y = ( blockIdx.y * blockDim.y + threadIdx.y) * GROUP_SIZE;

    

    for(int i=thread_position_x; i<thread_position_x+GROUP_SIZE; i++){
        if( i>= resX) break;

        for(int j=thread_position_y; j<thread_position_y+GROUP_SIZE; j++){
            if(j>=resY) break;

            float x = lowerX + (float)(i * stepX); //the actual position on the complex plane
            float y = lowerY + (float)(j * stepY);

            int idx = i + (j * resX);
            device_result[idx] = mandel(x, y, maxIterations);
        }
    }

    
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{   
    float stepX = (upperX - lowerX) / resX; //mean a stride of X-axis
    float stepY = (upperY - lowerY) / resY;

    /*
        Allocate Storage on Host & Device
    */

    int *host_result, *device_result; 
    int size = resX * resY * sizeof(int);

    size_t pitch;
    cudaHostAlloc(&host_result, size, cudaHostAllocDefault);
    cudaMallocPitch(&device_result, &pitch, resX*sizeof(int), resY);


    host_result = (int *)malloc(size);
    cudaMalloc(&device_result, size);

    /*
        Call Kernel Function
    */
    dim3 threads_per_block(BLOCK_SIZE/GROUP_SIZE, BLOCK_SIZE/GROUP_SIZE);
    dim3 num_blocks(resX / threads_per_block.x, resY / threads_per_block.y);
    mandelKernel<<<num_blocks, threads_per_block>>>(lowerX, lowerY, stepX, stepY, resX, resY, maxIterations, device_result, pitch);
    
    cudaMemcpy(host_result, device_result, size, cudaMemcpyDeviceToHost);
    memcpy(img, host_result, size);

    // Free allocated memory
    free(host_result);
    cudaFree(device_result);
}
