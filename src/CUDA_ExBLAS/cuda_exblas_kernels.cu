#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

#define WARP_COUNT 8
#define WARP_SIZE 32
#define BLOCK_SIZE (WARP_COUNT * WARP_SIZE)
#define MERGE_SUPERACCS_SIZE 128
#define PARTIAL_SUPERACCS_COUNT 512
#define MERGE_WORKGROUP_SIZE 64

#define BIN_COUNT      39
#define K              12                   // High-radix carry-save bits
#define digits         52
#define deltaScale     4503599627370496.0  // Assumes K>0
#define f_words        20
#define TSAFE           0


__device__ double TwoProductFMA(double a, double b, double *d) {
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

__device__ long int xadd(volatile long long int *sa, long long int x, unsigned char* of) {

    volatile unsigned long long int* cast_sa = (volatile unsigned long long int*)sa;

    unsigned long long int old, assumed;
    long long int y, z;

    old = *cast_sa;

    do{
        assumed = old;
        y = (long long int)assumed;
        z = y + x;
        // Because the bits does not change when we cast signed to unsigned.
        old = atomicCAS((unsigned long long int*)cast_sa, assumed, (unsigned long long int)z);
    }   while (old != assumed);

    *of = 0;
    if(x > 0 && y > 0 && z < 0)
        *of = 1;
    if(x < 0 && y < 0 && z > 0)
        *of = 1;

    return y;
}



////////////////////////////////////////////////////////////////////////////////
// Rounding functions
////////////////////////////////////////////////////////////////////////////////
__device__ double OddRoundSumNonnegative(double th, double tl) {
    union {
        double d;
        long long int l;
    } thdb;

    thdb.d = th + tl;
    // - if the mantissa of th is odd, there is nothing to do
    // - otherwise, round up as both tl and th are positive
    // in both cases, this means setting the msb to 1 when tl>0
    thdb.l |= (tl != 0.0);
    return thdb.d;
}

__device__ int Normalize(long long int *accumulator, int *imin, int *imax) {
    long long int carry_in = accumulator[*imin] >> digits;
    accumulator[*imin] -= carry_in << digits;
    int i;
    // Sign-extend all the way
    for (i = *imin + 1; i < BIN_COUNT; ++i) {
        accumulator[i] += carry_in;
        long long int carry_out = accumulator[i] >> digits;    // Arithmetic shift
        accumulator[i] -= (carry_out << digits);
        carry_in = carry_out;
    }
    *imax = i - 1;

    // Do not cancel the last carry to avoid losing information
    accumulator[*imax] += carry_in << digits;

    return carry_in < 0;
}

__device__ double Round(long long int *accumulator) {
    int imin = 0;
    int imax = 38;
    int negative = Normalize(accumulator, &imin, &imax);
    /*
    printf("Round() accumulator after Normalize:\n");
    for (int j = 0; j < BIN_COUNT; ++j) {
        printf("  acc[%2d] = %lld\n", j, accumulator[j]);
    }   
    */

    //Find leading word
    int i;
    //Skip zeroes
    for (i = imax; accumulator[i] == 0 && i >= imin; --i) {
    }
    if (negative) {
        //Skip ones
        for (; (accumulator[i] & ((1l << digits) - 1)) == ((1l << digits) - 1) && i >= imin; --i) {
        }
    }
    if (i < 0)
        //TODO: should we preserve sign of zero?
        return 0.0;

    long long int hiword = negative ? ((1l << digits) - 1) - accumulator[i] : accumulator[i];
    double rounded = (double) hiword;
    double hi = ldexp(rounded, (i - f_words) * digits);
    if (i == 0)
        return negative ? -hi : hi;  // Correct rounding achieved
    hiword -= (long long int) rint(rounded);
    double mid = ldexp((double) hiword, (i - f_words) * digits);

    //Compute sticky
    long long int sticky = 0;
    for (int j = imin; j != i - 1; ++j)
        sticky |= negative ? (1l << digits) - accumulator[j] : accumulator[j];

    long long int loword = negative ? (1l << digits) - accumulator[i - 1] : accumulator[i - 1];
    loword |= !!sticky;
    double lo = ldexp((double) loword, (i - 1 - f_words) * digits);

    //Now add3(hi, mid, lo)
    //No overlap, we have already normalized
    if (mid != 0)
        lo = OddRoundSumNonnegative(mid, lo);

    //Final rounding
    hi = hi + lo;
    return negative ? -hi : hi;
}


////////////////////////////////////////////////////////////////////////////////
// Main computation pass: compute partial superaccs
////////////////////////////////////////////////////////////////////////////////
__device__ void AccumulateWord(volatile long long int *sa, int i, long long int x) {
    // With atomic superacc updates
    // accumulation and carry propagation can happen in any order,
    // as long as addition is atomic
    // only constraint is: never forget an overflow bit
    unsigned char overflow;
    long long int carry = x;
    long long int carrybit;
    long long int oldword = xadd(&sa[i * WARP_COUNT], x, &overflow);

    // To propagate over- or underflow
    while (overflow) {
        // Carry or borrow
        // oldword has sign S
        // x has sign S
        // superacc[i] has sign !S (just after update)
        // carry has sign !S
        // carrybit has sign S
        carry = (oldword + carry) >> digits;    // Arithmetic shift
        bool s = oldword > 0;
        carrybit = (s ? 1l << K : -1l << K);

        // Cancel carry-save bits
        xadd(&sa[i * WARP_COUNT], (long long int) -(carry << digits), &overflow);
        if (TSAFE && (s ^ overflow))
            carrybit *= 2;
        carry += carrybit;

        ++i;
        if (i >= BIN_COUNT)
            return;
        oldword = xadd(&sa[i * WARP_COUNT], carry, &overflow);
    }
}

__device__ void Accumulate(volatile long long int *sa, double x) {
    if (x == 0)
        return;

    int e;
    frexp(x, &e);
    int exp_word = e / digits;  // Word containing MSbit
    int iup = exp_word + f_words;

    double xscaled = ldexp(x, -digits * exp_word);

    int i;
    for (i = iup; xscaled != 0; --i) {
        double xrounded = rint(xscaled);
        long long int xint = (long long int) xrounded;

        AccumulateWord(sa, i, xint);

        xscaled -= xrounded;
        xscaled *= deltaScale;
    }
}

__global__ void FinalReduceAndRound(double *d_Res, long long int *d_PartialSuperaccs, int block_count) {
    int tid = threadIdx.x;
    if (tid == 0) printf("[DEVICE] tid == 0 körs\n");
    if (tid < BIN_COUNT) {
        long long int sum = 0;
        for (int i = 0; i < block_count; ++i) {
            sum += d_PartialSuperaccs[i * BIN_COUNT + tid];
        }
        d_PartialSuperaccs[tid] = sum;
    }

    __syncthreads();

    if (tid == 0) {
        d_Res[0] = Round(d_PartialSuperaccs);
        printf("ExDOT dot result, in kernel: %.20e\n", d_Res[0]);
    }
}



__global__ void ExDOTComplete(long long int *d_PartialSuperaccs) {
    if(threadIdx.x == 0 && blockIdx.x == 0){
        printf("launched ExDOT complete\n");
    }
    
    unsigned int tid = threadIdx.x;
   
    
    if (tid < BIN_COUNT){
        long long int sum = 0;

        for (unsigned int i = 0; i < MERGE_SUPERACCS_SIZE; i++){
            sum += d_PartialSuperaccs[(blockIdx.x * MERGE_SUPERACCS_SIZE + i) * BIN_COUNT + tid];
        }
        d_PartialSuperaccs[blockIdx.x * BIN_COUNT + tid] = sum;
    }

    __syncthreads();

    if (tid == 0){
        int imin = 0;
        int imax = 38;
        Normalize(&d_PartialSuperaccs[blockIdx.x * BIN_COUNT], &imin, &imax);
    }

}


__global__ void ExDOT(long long int *d_PartialSuperaccs, double *d_a, double *d_b, unsigned int NbElements) {
    
    if(threadIdx.x == 0 && blockIdx.x == 0){
        printf("launched ExDOT\n");
    }
    unsigned int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ long long int l_sa[WARP_COUNT * BIN_COUNT];
    // pointer to the part of the shared (local) superaccumulator that its own warp owns.
    long long int *l_workingBase = l_sa + (tid & (WARP_COUNT - 1));

    //Initialize superaccs
    for(unsigned int i = 0; i < BIN_COUNT; i++){
        l_workingBase[i * WARP_COUNT] = 0;
    }
    __syncthreads();

    //TwoProductFMA - exakt multiplication
    //KnuthTwoSUM - exakt addition
    
    // FPE of size 4
    double a[4] = {0.0};

    double r, x, s;

    for (unsigned int pos = gid; pos < NbElements; pos += gridDim.x * blockDim.x) {

        r = 0.0;
        x = TwoProductFMA(d_a[pos], d_b[pos], &r);

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
        
       
    }
    //Flush FPEs to superaccs
    Accumulate(l_workingBase, a[0]);
    Accumulate(l_workingBase, a[1]);
    Accumulate(l_workingBase, a[2]);
    Accumulate(l_workingBase, a[3]);
    __syncthreads();

    //Merge sub-superaccs into work-group partial-accumulator
    if (tid < BIN_COUNT) {
        long long int sum = 0;
        
        for(unsigned int i = 0; i < WARP_COUNT; i++){
            sum += l_sa[tid * WARP_COUNT + i];
        }
        
        // one superacc per bin
        d_PartialSuperaccs[blockIdx.x * BIN_COUNT + tid] = sum;
    }

    __syncthreads();
    if (tid == 0){
        int imin = 0;
        int imax = 38;
        Normalize(&d_PartialSuperaccs[blockIdx.x * BIN_COUNT], &imin, &imax);
    }

}
    


extern "C" void launch_ExDOT(
   long long int *d_PartialSuperaccs, double *d_a, double *d_b, unsigned int NbElements){

    ExDOT<<<PARTIAL_SUPERACCS_COUNT, BLOCK_SIZE>>>(d_PartialSuperaccs, d_a, d_b, NbElements);
}

extern "C" void launch_ExDOTComplete(long long int *d_PartialSuperaccs){
    
    ExDOTComplete<<<PARTIAL_SUPERACCS_COUNT / MERGE_SUPERACCS_SIZE, MERGE_WORKGROUP_SIZE >>>(d_PartialSuperaccs);
}


extern "C" void launch_FinalReduceAndRound(double *d_Res, long long int *d_PartialSuperaccs){
    int block_count = PARTIAL_SUPERACCS_COUNT / MERGE_SUPERACCS_SIZE;
    FinalReduceAndRound<<<1, BIN_COUNT>>>(d_Res, d_PartialSuperaccs, block_count);
}
