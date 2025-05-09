#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

#define WARP_COUNT 8
#define WARP_SIZE 32
#define BLOCK_SIZE (WARP_COUNT * WARP_SIZE)

#define BIN_COUNT      39
#define K              12                   // High-radix carry-save bits
#define digits         52
#define deltaScale     4503599627370496.0  // Assumes K>0
#define f_words        20
#define TSAFE           0


double TwoProductFMA(double a, double b, double *d) {
    double p = a * b;
    *d = fma(a, b, -p);
    return p;
}

__device__ double KnuthTwoSum(double a, double b, double *s) {
    double r = a + b;
    double z = r - a;
    *s = (a - (r - z)) + (b - z);
    return r;
}



template <unsigned int blockSize>
__device__ void warpReduce(volatile double *sdata, unsigned int tid) {
    if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
    if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
    if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
    if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}

template <unsigned int blockSize>
__global__ void ExDOTComplete(double *g_idata, double *g_odata, unsigned int n) {
    extern __shared__ double sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize*2) + tid;
    unsigned int gridSize = blockSize*2*gridDim.x;
    sdata[tid] = 0;
    while (i < n) { sdata[tid] += g_idata[i] + g_idata[i+blockSize]; i += gridSize; }
    __syncthreads();
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
    if (tid < 32) warpReduce<BLOCK_SIZE>(sdata, tid);
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

template <unsigned int blockSize>
__global__ void ExDOT(double *d_a, double *d_b, double* result, unsigned int n) {

    // 
    unsigned int tid = threadIdx.x;
    __shared__ long long int l_sa[WARP_COUNT * BIN_COUNT];
    // pointer to the part of the shared (local) superaccumulator that its own warp owns.
    long long int *l_workingBase = l_sa + (tid & (WARP_COUNT - 1));

    //Initialize superaccs
    for(unsigned int = 0; i < BIN_COUNT; i++){
        l_workingBase[i * WARP_COUNT] = 0;
    }
    __syncthreads();

    //TwoProductFMA - exakt multiplication
    //KnuthTwoSUM - exakt addition
    
    // FPE of size 4
    double a[4] = {0.0};

    //extern __shared__ double sdata[];
    unsigned int i = blockIdx.x*(blockSize*2) + tid;
    unsigned int gridSize = blockSize*2*gridDim.x;

    //sdata[tid] = 0;

    double r, x, s;

    while (i < n){

        r = 0.0;
        x = TwoProductFMA(d_a[i], d_b[i], &r);

        a[0] = KnuthTwoSum(a[0], x, &s);
        x  = s;
        if (x != 0.0){
            a[1] = KnuthTwoSum(a[1], x, &s);
            x = s;
            if (x != 0.0){
                a[2] = KnuthTwoSum(a[2], x, &s);
                x = s;
                if(x != 0.0){
                    a[3] = KnuthTwoSum(a[3], x, &s);
                    x = s;
                }
            }
        }
        if (x != 0.0){
            Accumulate(l_workingBase, x);
            //Flush FPEs to superaccs
            Accumulate(l_workingBase, a[0]);
            Accumulate(l_workingBase, a[1]);
            Accumulate(l_workingBase, a[2]);
            Accumulate(l_workingBase, a[3]);
            a[0] = 0.0;
			a[1] = 0.0;
			a[2] = 0.0;
			a[3] = 0.0;
        }

        if (r != 0.0){
            a[1] = KnuthTwoSum(a[1], r, &s);
            r = s;
            if (r != 0.0){
                a[2] = KnuthTwoSum(a[2], r, &s);
                r = s;
                if (r != 0.0){
                    a[3] = KnuthTwoSum(a[3], r, &s);
                    r = s;
                }
            }

        }

        if (r != 0.0){
            Accumulate(l_workingBase, r);
            //Flush FPEs to superaccs
            Accumulate(l_workingBase, a[0]);
			Accumulate(l_workingBase, a[1]);
			Accumulate(l_workingBase, a[2]);
			Accumulate(l_workingBase, a[3]);
            a[0] = 0.0;
			a[1] = 0.0;
			a[2] = 0.0;
			a[3] = 0.0;

        }


        
        r = 0.0;
        x = TwoProductFMA(d_a[i+blockSize], d_b[i+blockSize], &r);
       
        a[0] = KnuthTwoSum(a[0], x, &s);
        x  = s;
        if (x != 0.0){
            a[1] = KnuthTwoSum(a[1], x, &s);
            x = s;
            if (x != 0.0){
                a[2] = KnuthTwoSum(a[2], x, &s);
                x = s;
                if(x != 0.0){
                    a[3] = KnuthTwoSum(a[3], x, &s);
                    x = s;
                }
            }
        }
        if (x != 0.0){
            Accumulate(l_workingBase, x);
            //Flush FPEs to superaccs
            Accumulate(l_workingBase, a[0]);
            Accumulate(l_workingBase, a[1]);
            Accumulate(l_workingBase, a[2]);
            Accumulate(l_workingBase, a[3]);
            a[0] = 0.0;
			a[1] = 0.0;
			a[2] = 0.0;
			a[3] = 0.0;
        }

        if (r != 0.0){
            a[1] = KnuthTwoSum(a[1], r, &s);
            r = s;
            if (r != 0.0){
                a[2] = KnuthTwoSum(a[2], r, &s);
                r = s;
                if (r != 0.0){
                    a[3] = KnuthTwoSum(a[3], r, &s);
                    r = s;
                }
            }

        }

        if (r != 0.0){
            Accumulate(l_workingBase, r);
            //Flush FPEs to superaccs
            Accumulate(l_workingBase, a[0]);
			Accumulate(l_workingBase, a[1]);
			Accumulate(l_workingBase, a[2]);
			Accumulate(l_workingBase, a[3]);
            a[0] = 0.0;
			a[1] = 0.0;
			a[2] = 0.0;
			a[3] = 0.0;

        }

        i += gridSize;
    }

    //Flush FPEs to superaccs
    Accumulate(l_workingBase, a[0]);
    Accumulate(l_workingBase, a[1]);
    Accumulate(l_workingBase, a[2]);
    Accumulate(l_workingBase, a[3]);
    __syncthreads();

    //Merge sub-superaccs into work-group partial-accumulator

    
    if (blockSize >= 512) { 
        if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); 
    }
    if (blockSize >= 256) { 
        if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); 
    }
    if (blockSize >= 128) { 
        if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); 
    }
    if (tid < 32) warpReduce<BLOCK_SIZE>(sdata, tid);
    if (tid == 0) result[blockIdx.x] = sdata[0];

    




    if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
    if (tid < 32) warpReduce<BLOCK_SIZE>(sdata, tid);
    if (tid == 0) result[blockIdx.x] = sdata[0];
    }


extern "C" void launch_ExDOT(
    long long int *d_PartialSuperaccs,
    double *d_a,
    double *d_b,
    const unsigned int NbElements
){

    ExDOT<<<>>>(d_PartialSuperaccs, d_a, d_b, NbElements);
}

extern "C" void launch_ExDOTComplete(
    double *d_Res,
    long long int *d_PartialSuperaccs,
    unsigned int PartialSuperaccusCount
){
    
    ExDOTComplete<<<>>>(d_Res, d_PartialSuperaccs, PartialSuperaccusCount);
}
