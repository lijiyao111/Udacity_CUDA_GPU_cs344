//Udacity HW 4
//Radix Sorting

#include "reference_calc.cpp"
#include "utils.h"

/* Red Eye Removal
   ===============
   
   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

   Note: ascending order == smallest to largest

   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.

   Implementing Parallel Radix Sort with CUDA
   ==========================================

   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.

   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there

   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.

 */
#include <stdio.h>

__host__ __device__
void swap(int & a, int &b){
    a=a^b;
    b=a^b;
    a=b^a;
}

// __device__ unsigned int d_bins[2];

__global__
void print_bins(unsigned int* d_bins, int i){
    printf("Device: %d %d total %d where: %d\n",d_bins[0],d_bins[1], d_bins[0]+d_bins[1],i);
}

// Scan, limited to 1 block, upto 1024 threads; 
__global__ 
void scan(unsigned int *g_data, unsigned int * g_intermediate, int n, int flag)  {
    // flag =0 inclusive; flag =1 Exclusive
    extern __shared__ unsigned int temp[]; // allocated on invocation  
    int gid=threadIdx.x + blockIdx.x*blockDim.x;
    int Ndim=blockDim.x;
    int thid = threadIdx.x;
    int ln= (blockDim.x*(blockIdx.x+1)>n)? (n - blockDim.x*blockIdx.x) : blockDim.x;
    int pin=0,pout=1;
    unsigned int data_end;

    // if(threadIdx.x==0) printf("in Scan %d %d\n", d_bins[0],d_bins[1]);

    // Load input into shared memory.  
     // This is exclusive scan, so shift right by one  
     // and set first element to 0  
    if (gid>=n)
        return;     
        // temp[thid+pin*Ndim]=g_idata[gid]; // Inclusive
    if (flag ==0){
        temp[thid+pout*Ndim]=g_data[gid]; // Inclusive
    }else{
        temp[thid+pout*Ndim] = (thid > 0) ? g_data[gid-1] : 0;   // Exclusive
        data_end=g_data[gid];
    }

    // if(threadIdx.x==0) printf("in Scan %d %d\n", d_bins[0],d_bins[1]);

    __syncthreads();  
    // printf("%d\n",thid);
    for (int offset = 1; offset < ln; offset *= 2)  
    { 

      swap(pin,pout);
      if (thid >= offset)  
        temp[pout*Ndim+thid] = temp[pin*Ndim+thid]+ temp[pin*Ndim+thid - offset];  
      else  
        temp[pout*Ndim+thid] = temp[pin*Ndim+thid];  

    __syncthreads();  
    }  

    g_data[gid] = temp[pout*Ndim+thid]; // write output 

    if(thid==ln-1){
        if(flag == 0){
            g_intermediate[blockIdx.x]=temp[pout*Ndim+thid];
        } else{
            g_intermediate[blockIdx.x]=temp[pout*Ndim+thid]+data_end; // Exclusive
        }
    }

 }

 __global__
 void scan_extra(unsigned int *g_io, unsigned int * g_intermediate, int n){
    int gid=threadIdx.x + blockIdx.x*blockDim.x;
    int interid=blockIdx.x;
    // int thid = threadIdx.x;
    if(gid<n)
        g_io[gid] +=g_intermediate[interid];
 }

 void scan_large(unsigned int * d_in,const int N){
    unsigned int * d_intermediate;
    int Nthread=1024;
    int Nblock=(N+Nthread-1)/Nthread;
    int Nblock_s=Nblock;
    int flag =1; //inclusive; flag =1 Exclusive

    // h_intermediate=(unsigned int *) malloc(Nblock*sizeof(unsigned int));
    cudaMalloc(&d_intermediate,Nblock*sizeof(unsigned int));

    scan<<<Nblock,Nthread,2*Nthread*sizeof(unsigned int)>>>(d_in,d_intermediate,N, flag);

    Nthread=Nblock;
    Nblock=1;
    flag =1; //inclusive; flag =1 Exclusive
    unsigned int * d_junk;
    cudaMalloc(&d_junk,Nblock*sizeof(unsigned int));

    scan<<<Nblock,Nthread,2*Nthread*sizeof(unsigned int)>>>(d_intermediate,d_junk,Nthread,flag);

    Nthread=1024;
    Nblock=(N+Nthread-1)/Nthread;
    scan_extra<<<Nblock,Nthread>>>(d_in,d_intermediate,N);

    cudaFree(d_intermediate); cudaFree(d_junk);
 }


__global__
void histogram_kernel(unsigned int pass,
                      unsigned int * d_bins,
                      unsigned const int*  d_input, 
                      const int size) {  
    int gid = threadIdx.x + blockDim.x * blockIdx.x;
    if(gid >= size)
        return;

    // reset_hist(d_bins);
    unsigned int one = 1;
    int bin = ((d_input[gid] & (one<<pass)) == (one<<pass)) ? 1 : 0;
    if(bin) 
         atomicAdd(&d_bins[1], 1);
    else
         atomicAdd(&d_bins[0], 1);
}

__global__
void digit_identify(unsigned const int * d_input, 
                unsigned int * d_out,const int N, const int pass, int flag=0){
    // flag == 0 for 0, or, for 1
    int gid = threadIdx.x + blockDim.x * blockIdx.x;
    if(gid < N){
        unsigned int one = 1;
        unsigned int label = ((d_input[gid] & (one<<pass)) == (one<<pass)) ? flag : 1-flag;

        d_out[gid]=label;
    }

}

// __global__
// void move(unsigned int *d_output,unsigned const int *d_input,unsigned const int *d_digitloc,unsigned const int *d_bins,
//     const int N, const int pass, int flag=0){
//         // flag == 0 for 0, or, for 1
//     int gid = threadIdx.x + blockDim.x * blockIdx.x;
//     if(gid >= N)
//         return;
//     unsigned int one = 1;
//     int bin = ((d_input[gid] & (one<<pass)) == (one<<pass)) ? flag : 1-flag;
//     // printf("Here %d %d %d\n", bin, d_input[gid],flag);
//     if(bin) {
//         int newloc=d_digitloc[gid]+d_bins[flag];
//         // printf("Here %d %d %d\n", bin, d_input[gid],flag);
//         // printf("Move  %d %d %d %d\n",newloc,d_input[gid], d_bins[flag], flag2);
//         // std::cout<<"Move "<< newloc<<d_input[gid]<<d_bins[flag]<<std::endl;
//         d_output[newloc]=d_input[gid];
//     }
// }

__global__
void move(unsigned int *d_output, unsigned int *d_output_pos,unsigned const int *d_input, 
            unsigned const int* d_input_pos, unsigned const int *d_digitloc,unsigned const int *d_bins,
            const int N, const int pass, int flag=0){
        // flag == 0 for 0, or, for 1
    int gid = threadIdx.x + blockDim.x * blockIdx.x;
    if(gid >= N)
        return;
    unsigned int one = 1;
    int bin = ((d_input[gid] & (one<<pass)) == (one<<pass)) ? flag : 1-flag;
    // printf("Here %d %d %d\n", bin, d_input[gid],flag);
    if(bin) {
        int newloc=d_digitloc[gid]+d_bins[flag];
        d_output[newloc]=d_input[gid];
        d_output_pos[newloc]=d_input_pos[gid];
    }
}

__global__
void print_digit(unsigned int* d_digitloc, int n){
    printf("Here: ");
    for(int i=0;i<n;++i){
        printf("%d ",d_digitloc[i]);
    }
    printf("\n");
}

__global__
void reset_bins(unsigned int *bin){
    bin[0]=0;
    bin[1]=0;
}


void radix_sort(unsigned const int * d_input_const,  unsigned const int * d_input_pos_const, 
                unsigned int * d_out, unsigned int * d_out_pos, const int N){
    unsigned int * d_digitloc;//,* h_digitloc;
    int sizeN= N * sizeof(unsigned int);

    // h_digitloc=(unsigned int *) malloc(sizeN);
    cudaMalloc(&d_digitloc, sizeN);

    unsigned int * d_input;
    cudaMalloc(&d_input,sizeN);
    cudaMemcpy(d_input,d_input_const,sizeN,cudaMemcpyDeviceToDevice);

    unsigned int* d_input_pos;
    cudaMalloc(&d_input_pos,sizeN);
    cudaMemcpy(d_input_pos,d_input_pos_const,sizeN,cudaMemcpyDeviceToDevice);

    unsigned int * d_bins;
    int Nblock, Nthread;

    cudaMalloc(&d_bins, 2*sizeof(unsigned int));
    unsigned int h_bins[2]={0};
    cudaMemcpy(d_bins,h_bins,2*sizeof(unsigned int),cudaMemcpyHostToDevice);


    for (int pass=0;pass<32;pass++){
        Nthread=1024;
        Nblock=(N+Nthread-1)/Nthread;



        histogram_kernel<<<Nblock,Nthread>>>(pass, d_bins, d_input, N);

        scan_large(d_bins, 2);


        Nthread=1024;
        Nblock=(N+Nthread-1)/Nthread;

        digit_identify<<<Nthread,Nblock>>>(d_input,  d_digitloc, N, pass, 0);
        scan_large(d_digitloc, N);
        move<<<Nthread,Nblock>>>(d_out,d_out_pos,d_input,d_input_pos, d_digitloc,d_bins,N,pass,0);


        digit_identify<<<Nthread,Nblock>>>(d_input,  d_digitloc, N, pass, 1);
        scan_large(d_digitloc, N);
        move<<<Nthread,Nblock>>>(d_out,d_out_pos,d_input,d_input_pos, d_digitloc,d_bins,N,pass,1);

        cudaMemcpy(d_input,d_out,sizeN,cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_input_pos,d_out_pos,sizeN,cudaMemcpyDeviceToDevice);

    }

    cudaFree(d_digitloc);  cudaFree(d_bins);
    cudaFree(d_input); cudaFree(d_input_pos);
    // free(h_digitloc);
}

void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{

radix_sort(d_inputVals, d_inputPos ,d_outputVals,d_outputPos, numElems);
 
}
