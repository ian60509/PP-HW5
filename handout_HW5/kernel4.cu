#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE  16

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

__global__ void mandelKernel(float lowerX, float lowerY, float stepX, float stepY, int resX, int resY, int maxIterations, int *device_result) {
    // To avoid error caused by the floating number, use the following pseudo code
    //
    // float x = lowerX + thisX * stepX;
    // float y = lowerY + thisY * stepY;
    int thread_position_x = blockIdx.x * blockDim.x + threadIdx.x; //threadIdx is the threadID in this specific block
    int thread_position_y = blockIdx.y * blockDim.y + threadIdx.y;

    float x = lowerX + (float)(thread_position_x * stepX);
    float y = lowerY + (float)(thread_position_y * stepY);

    int idx = thread_position_x + (thread_position_y * resX);
    device_result[idx] = mandel(x, y, maxIterations);
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


    host_result = (int *)malloc(size);
    cudaMalloc(&device_result, size);

    /*
        Call Kernel Function
    */
    dim3 threads_per_block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 num_blocks(resX / threads_per_block.x, resY / threads_per_block.y);
    mandelKernel<<<num_blocks, threads_per_block>>>(lowerX, lowerY, stepX, stepY, resX, resY, maxIterations, device_result);
    
    cudaMemcpy(host_result, device_result, size, cudaMemcpyDeviceToHost);
    memcpy(img, host_result, size);

    // Free allocated memory
    free(host_result);
    cudaFree(device_result);
}
