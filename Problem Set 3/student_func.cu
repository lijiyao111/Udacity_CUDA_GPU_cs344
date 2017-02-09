/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include "utils.h"
#include <stdio.h>

// Histogram calculation using atomic add
__global__
void hist_atomic(const float * lum, const float d_minlum, const float d_maxlum, unsigned int * hist,
  const int n,const int numBins){
    int id=blockIdx.x*blockDim.x+threadIdx.x;
    float lumRange=d_maxlum-d_minlum;//+1e-6f;
    // if(threadIdx.x==0){
    //  printf("%f %f %f\n",*d_minlum,*d_maxlum,lumRange);
    //  printf("%d %d\n",blockDim.x,blockIdx.x);
    // }
    if(id<n){
        int id_h=(lum[id] - d_minlum) / lumRange * (numBins);
        // hist[id_h] +=1;
        atomicAdd(& hist[id_h], 1); 
    }
}

// Scan, limited to 1 block, upto 1024 threads; 
__global__ 
void scan(unsigned int *g_odata, unsigned int *g_idata, int n)  {
  extern __shared__ float temp[]; // allocated on invocation  
   int thid = threadIdx.x;  
  int pout = 0, pin = 1;  
  // Load input into shared memory.  
   // This is exclusive scan, so shift right by one  
   // and set first element to 0  
  temp[pout*n + thid] = (thid > 0) ? g_idata[thid-1] : 0;  // Exclusive scan
  // temp[pout*n + thid]=g_idata[thid]; // Inclusive

  __syncthreads();  
  for (int offset = 1; offset < n; offset *= 2)  
  {  
    pout = 1 - pout; // swap double buffer indices  
    pin = 1 - pout;  
    if (thid >= offset)  
      temp[pout*n+thid] = temp[pin*n+thid] + temp[pin*n+thid - offset];  
    else  
      temp[pout*n+thid] = temp[pin*n+thid];  
    __syncthreads();  
  }  
  g_odata[thid] = temp[pout*n+thid]; // write output  
 }


// Reduce
__global__ 
void reduce_kernel(float * d_out, const float * d_in, int n, int op)
{
    // sdata is allocated in the kernel call: 3rd arg to <<<b, t, shmem>>>
    extern __shared__ float sdata[];

    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    int tid  = threadIdx.x;

    // load shared mem from global mem
    if(myId<n){
      sdata[tid] = d_in[myId];
    }
    __syncthreads();            // make sure entire block is loaded!

    // do reduction in shared mem
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if ((myId+s<n) && (tid < s) )
        {
          if (op==0){
        sdata[tid]=min(sdata[tid], sdata[tid+s]);
          }else if (op==1){
        sdata[tid]=max(sdata[tid], sdata[tid+s]);
          }else{
            sdata[tid] += sdata[tid + s];
          }
            // 
            
        }
        __syncthreads();        // make sure all adds at one stage are done!
    }

    // only thread 0 writes result for this block back to global mem
    if (tid == 0)
    {
        d_out[blockIdx.x] = sdata[0];
    }
}

__global__
void reset_hist(unsigned int * hist, int N){
  for (int i=0;i<N;i++){
    hist[i]=0;
  }

}


__host__ __device__
unsigned int round2power(unsigned int v){
  // unsigned int v; // compute the next highest power of 2 of 32-bit v
  v--;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  v++;
  return v;
}

void reduce(const float* const d_in,float & h_reduce, int h_op, const size_t N){

  unsigned int Nblock, Nthread;
  float *d_reduce;
  cudaMalloc(&d_reduce,sizeof(float));
  // cudaMalloc(&d_maxlum,sizeof(float));

  // int h_op=0;  // 0 Min, 1 Max, else Sum
  Nthread=512;
  Nblock=(N+Nthread-1)/Nthread;
  int Nblock_s=Nblock;
  Nblock=round2power(Nblock_s);

  float *d_intermediate;
  cudaMalloc(&d_intermediate,Nblock*sizeof(float));

  cudaMemcpy(d_reduce,&h_reduce,sizeof(float),cudaMemcpyHostToDevice);
  // cudaMemcpy(d_maxlum,&h_maxlum,sizeof(float),cudaMemcpyHostToDevice);
  reduce_kernel<<<Nblock, Nthread, Nthread * sizeof(float)>>>(d_intermediate,d_in,N,h_op);

  Nthread=Nblock;
  Nblock=1;
  // printf("Block %d %d\n",Nthread,Nblock);
  reduce_kernel<<<Nblock, Nthread, Nthread * sizeof(float)>>>(d_reduce,d_intermediate,Nblock_s,h_op);

  // print_vals2<<<1,1>>>(d_minlum,d_maxlum);

  cudaMemcpy(&h_reduce, d_reduce, sizeof(float), cudaMemcpyDeviceToHost);
  // cudaMemcpy(&h_maxlum, d_maxlum, sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_intermediate);
  cudaFree(d_reduce);

}

// __global__
// void print_arr(unsigned int * arr, int N){
//   for(int i=0; i<N; ++i){
//     printf("%u ",arr[i]);
//   }
//   printf("\n");
// }


void create_cdf(const float* const d_lumin, unsigned int* const d_cdf,
              float & h_minlum, float  &h_maxlum, const size_t N, const size_t N_bin){

  int Nblock, Nthread;


  Nthread=512;
  Nblock=(N+Nthread-1)/Nthread;


  int sizeN=N*sizeof(unsigned int);
  int sizeN_bin=N_bin*sizeof(unsigned int);


  unsigned int * d_hist, *dc_cdf;

  cudaMalloc(&d_hist,sizeN_bin);
  cudaMalloc(&dc_cdf,sizeN_bin);

  reset_hist<<<1,1>>>(d_hist,N_bin);

  reduce(d_lumin, h_minlum, 0,N);
  reduce(d_lumin, h_maxlum, 1,N);

  Nthread=512;
  Nblock=(N+Nthread-1)/Nthread;
  hist_atomic<<<Nblock,Nthread>>>(d_lumin, h_minlum,h_maxlum, d_hist,N, N_bin);


  Nblock=1;
  Nthread=N_bin;
  scan<<<Nblock,Nthread,sizeN_bin*2>>>(d_cdf,d_hist,N_bin);
  // print_arr<<<1,1>>>(d_cdf,N_bin);

  cudaFree(d_hist); cudaFree(dc_cdf);

}



void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  //TODO
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */

       printf("Num %d %d %d\n", numBins, numRows, numCols);
       printf("val %f %f\n",min_logLum, max_logLum);


       create_cdf(d_logLuminance, d_cdf,
              min_logLum, max_logLum, numRows*numCols, numBins);
       printf("val %f %f\n",min_logLum, max_logLum);


}
