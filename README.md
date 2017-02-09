# Udacity class Intro to Parallel Programming by Nvidia
> CUDA GPU computing. This class talks lots about parallel algorithms but not much about the CUDA syntax. 
> The class is [here](https://www.udacity.com/course/intro-to-parallel-programming--cs344) on Udacity website.
> 
> Class projects assignments are on [Github](https://github.com/udacity/cs344).
> 
> Here is my solutions for the first 4 projects (of total 6 projects).
> 
> Also check my basic CUDA code [here](https://github.com/lijiyao111/CUDA_C).

## Project 1
> Map

transfer RGB colorful image into gray image. Very easy. 

## Project 2
> 2D stencil operation. 

Blur an image. Relatively easy.

## Project 3
> Histogram, Reduce, Scan

Tone mapping of image. Relatively difficult. 

Reduce with multiple blocks is not quite easy. 

This project only use scan with 1 block due to small number of elements to scan. However, the scan code I studied from Nvidia website (http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html) has bug and does not work correctly. How could you do this ```temp[pout*n+thid] += temp[pin*n+thid - offset]; ```? It should be ```temp[pout*n+thid] = temp[pout*n+thid] + temp[pin*n+thid - offset]; ```.

My Reduce, Scan, Histogram can deal with arbitrary number of input (Scan can only have 1 block in this project, see the next project for scan with large number of input).

## Project 4
> Histogram, Compact, large Scan, Radix sort

Red eye removal. Difficult. Sorting with CUDA is not as simple as sorting with CPU in seriel. 

Scan code has been modified to allow arbitrary number of input. 