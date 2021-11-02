/* Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * Greyscale Conversion
 *
 * This sample is a very basic sample that implements color to greyscale conversion.
 * It is the same as the sample illustrating Chapter 3
 * of the programming guide with some additions like error checking.
 */

#include <stdio.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

#include <helper_cuda.h>
/**
 * CUDA Kernel Device code
 *
 * Computes the greyscale conversion. The 2 vectors have the same
 * number of elements numElements.
 */
__global__ void greyScale(const float3 *in, float *out, int width, int height)
{
  int col = blockDim.x * blockIdx.x + threadIdx.x;
  int row = blockDim.y * blockIdx.y + threadIdx.y;

  if (col < width && row < height)
  {
    int i = row * width + col;
    out[i] = in[i].x * 0.21 + in[i].y * 0.72 + in[i].z * 0.07;
  }
}

/**
 * Host main routine
 */
int main(void)
{
  // Error code to check return values for CUDA calls
  cudaError_t err = cudaSuccess;

  // Print the image size to be used
  int width = 1920;
  int height = 1080;
  printf("[Greyscale conversion of %dx%d image]\n", width, height);

  // Allocate the host input vector
  size_t sizeIn = width * height * sizeof(float3);
  float3 *h_in = (float3 *)malloc(sizeIn);

  // Allocate the host output vector
  size_t sizeOut = width * height * sizeof(float);
  float *h_out = (float *)malloc(sizeOut);

  // Verify that allocations succeeded
  if (h_in == NULL || h_out == NULL)
  {
    fprintf(stderr, "Failed to allocate host vectors!\n");
    exit(EXIT_FAILURE);
  }

  // Initialize the host input vectors
  for (int i = 0; i < width * height; ++i)
  {
    h_in[i].x = rand() / (float)RAND_MAX;
    h_in[i].y = rand() / (float)RAND_MAX;
    h_in[i].z = rand() / (float)RAND_MAX;
  }

  // Allocate the device input vector
  float3 *d_in = NULL;
  err = cudaMalloc((void **)&d_in, sizeIn);

  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to allocate device input vector (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Allocate the device output vector
  float *d_out = NULL;
  err = cudaMalloc((void **)&d_out, sizeOut);

  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to allocate device output vector (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Copy the host input vector in host memory to the device input vector in device memory
  printf("Copy input data from the host memory to the CUDA device\n");
  err = cudaMemcpy(d_in, h_in, sizeIn, cudaMemcpyHostToDevice);

  if (err != cudaSuccess)
  {
    fprintf(stderr,"Failed to copy input vector from host to device (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Launch the Greyscale CUDA Kernel
  dim3 threadsPerBlock(16, 16, 1);
  dim3 blocksPerGrid(ceil(width/16.0),ceil(height/16.0),1);
  printf("CUDA kernel launch with %dx%d blocks of %dx%d threads\n",
         blocksPerGrid.x, blocksPerGrid.y,
         threadsPerBlock.x, threadsPerBlock.y);

  greyScale<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, width, height);
  err = cudaGetLastError();

  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to launch greyScale kernel (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Copy the device result vector in device memory to the host result vector
  // in host memory.
  printf("Copy output data from the CUDA device to the host memory\n");
  err = cudaMemcpy(h_out, d_out, sizeOut, cudaMemcpyDeviceToHost);

  if (err != cudaSuccess)
  {
    fprintf(stderr,"Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Verify that the greyscale conversion is correct
  for (int i = 0; i < width * height; ++i)
  {
    if (fabs(h_in[i].x * 0.21 + h_in[i].y * 0.72 + h_in[i].z * 0.07 - h_out[i]) > 1e-5)
    {
      fprintf(stderr, "Result verification failed at element %d!\n", i);
      exit(EXIT_FAILURE);
    }
  }

  printf("Test PASSED\n");

  // Free device global memory
  err = cudaFree(d_in);

  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to free input device vector (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  err = cudaFree(d_out);

  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to free output device vector (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Free host memory
  free(h_in);
  free(h_out);

  printf("Done\n");
  return 0;
}
