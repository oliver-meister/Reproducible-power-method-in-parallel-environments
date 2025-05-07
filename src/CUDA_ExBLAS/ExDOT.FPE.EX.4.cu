
#define BIN_COUNT      39
#define K              12                   // High-radix carry-save bits
#define digits         52
#define deltaScale     4503599627370496.0  // Assumes K>0
#define f_words        20
#define TSAFE           0
#define EARLY_EXIT      1

#define WORKGROUP_SIZE 256
#define WARP_COUNT 16
#define WARP_SIZE 16
#define MERGE_WORKGROUP_SIZE 64
#define MERGE_SUPERACCS_SIZE 128

#define PARTIAL_SUPERACCS_COUNT 512


////////////////////////////////////////////////////////////////////////////////
// Auxiliary functions
////////////////////////////////////////////////////////////////////////////////
double TwoProductFMA(double a, double b, double *d) {
    double p = a * b;
    *d = fma(a, b, -p);
    return p;
}

double KnuthTwoSum(double a, double b, double *s) {
    double r = a + b;
    double z = r - a;
    *s = (a - (r - z)) + (b - z);
    return r;
}

// signedcarry in {-1, 0, 1}
long xadd(volatile long *sa, long x, unsigned char* of) {
    // OF and SF  -> carry=1
    // OF and !SF -> carry=-1
    // !OF        -> carry=0
    long y = atom_add(sa, x);
    long z = y + x; // since the value sa->superacc[i] can be changed by another work item

    // TODO: cover also underflow
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
double OddRoundSumNonnegative(double th, double tl) {
    union {
        double d;
        long l;
    } thdb;

    thdb.d = th + tl;
    // - if the mantissa of th is odd, there is nothing to do
    // - otherwise, round up as both tl and th are positive
    // in both cases, this means setting the msb to 1 when tl>0
    thdb.l |= (tl != 0.0);
    return thdb.d;
}

int Normalize(long *accumulator, int *imin, int *imax) {
    long carry_in = accumulator[*imin] >> digits;
    accumulator[*imin] -= carry_in << digits;
    int i;
    // Sign-extend all the way
    for (i = *imin + 1; i < BIN_COUNT; ++i) {
        accumulator[i] += carry_in;
        long carry_out = accumulator[i] >> digits;    // Arithmetic shift
        accumulator[i] -= (carry_out << digits);
        carry_in = carry_out;
    }
    *imax = i - 1;

    // Do not cancel the last carry to avoid losing information
    accumulator[*imax] += carry_in << digits;

    return carry_in < 0;
}

double Round(long *accumulator) {
    int imin = 0;
    int imax = 38;
    int negative = Normalize(accumulator, &imin, &imax);

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

    long hiword = negative ? ((1l << digits) - 1) - accumulator[i] : accumulator[i];
    double rounded = (double) hiword;
    double hi = ldexp(rounded, (i - f_words) * digits);
    if (i == 0)
        return negative ? -hi : hi;  // Correct rounding achieved
    hiword -= (long) rint(rounded);
    double mid = ldexp((double) hiword, (i - f_words) * digits);

    //Compute sticky
    long sticky = 0;
    for (int j = imin; j != i - 1; ++j)
        sticky |= negative ? (1l << digits) - accumulator[j] : accumulator[j];

    long loword = negative ? (1l << digits) - accumulator[i - 1] : accumulator[i - 1];
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
void AccumulateWord(volatile long *sa, int i, long x) {
    // With atomic superacc updates
    // accumulation and carry propagation can happen in any order,
    // as long as addition is atomic
    // only constraint is: never forget an overflow bit
    uchar overflow;
    long carry = x;
    long carrybit;
    long oldword = xadd(&sa[i * WARP_COUNT], x, &overflow);

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
        xadd(&sa[i * WARP_COUNT], (long) -(carry << digits), &overflow);
        if (TSAFE && (s ^ overflow))
            carrybit *= 2;
        carry += carrybit;

        ++i;
        if (i >= BIN_COUNT)
            return;
        oldword = xadd(&sa[i * WARP_COUNT], carry, &overflow);
    }
}

void Accumulate(volatile long *sa, double x) {
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
        long xint = (long) xrounded;

        AccumulateWord(sa, i, xint);

        xscaled -= xrounded;
        xscaled *= deltaScale;
    }
}


__global__ void ExDOT(
    long *d_PartialSuperaccs,
    double *d_a,
    double *d_b,
    const unsigned int NbElements
) {
    
    __shared__ long l_sa[WARP_COUNT * BIN_COUNT];
    long *l_workingBase = &l_sa[threadIdx.x & (WARP_COUNT - 1)];

    //Initialize superaccs
    for (uint i = 0; i < BIN_COUNT; i++)
        l_workingBase[i * WARP_COUNT] = 0;
    __syncthreads();

    //Read data from global memory and scatter it to sub-superaccs
    double a[4] = {0.0};

    
	for(unsigned int pos = blockIdx.x * blockDim.x + threadIdx.x; pos < NbElements; pos += gridDim.x * blockDim.x){
		double r = 0.0;
			double x = TwoProductFMA(d_a[pos], d_b[pos], &r);

		double s;
		a[0] = KnuthTwoSum(a[0], x, &s);
		x = s;
		if (x != 0.0) {
			a[1] = KnuthTwoSum(a[1], x, &s);
			x = s;
			if (x != 0.0) {
				a[2] = KnuthTwoSum(a[2], x, &s);
				x = s;
				if (x != 0.0) {
					a[3] = KnuthTwoSum(a[3], x, &s);
					x = s;
				}
			}
		}
		if(x != 0.0) {
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

		if(r != 0.0) {
			/*a[0] = KnuthTwoSum(a[0], r, &s);
			r = s;
			if(r != 0.0) {*/
			a[1] = KnuthTwoSum(a[1], r, &s);
			r = s;
			if (r != 0.0) {
				a[2] = KnuthTwoSum(a[2], r, &s);
				r = s;
				if (r != 0.0) {
					a[3] = KnuthTwoSum(a[3], r, &s);
					r = s;
				}
			}
			//}
			if(r != 0.0) {
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
	}
    //Flush FPEs to superaccs
    Accumulate(l_workingBase, a[0]);
    Accumulate(l_workingBase, a[1]);
    Accumulate(l_workingBase, a[2]);
    Accumulate(l_workingBase, a[3]);
    __syncthreads();

    //Merge sub-superaccs into work-group partial-accumulator
    unsigned int pos = threadIdx.x;
    if (pos < BIN_COUNT) {
        long sum = 0;

        for(unsigned int i = 0; i < WARP_COUNT; i++)
            sum += l_sa[pos * WARP_COUNT + i];

        d_PartialSuperaccs[blockIdx.x * BIN_COUNT + pos] = sum;
    }

    __syncthreads();
    if (pos == 0) {
        int imin = 0;
        int imax = 38;
        Normalize(&d_PartialSuperaccs[blockIdx.x * BIN_COUNT], &imin, &imax);
    }
}

////////////////////////////////////////////////////////////////////////////////
// Merging
////////////////////////////////////////////////////////////////////////////////

__global__ void ExDOTComplete(
    double *d_Res,
    long *d_PartialSuperaccs,
    unsigned int PartialSuperaccusCount
) {
    unsigned int lid = threadIdx.x;
    unsigned int gid = blockIdx.x;

    if (lid < BIN_COUNT) {
        long sum = 0;

        for(unsigned int i = 0; i < MERGE_SUPERACCS_SIZE; i++)
            sum += d_PartialSuperaccs[(gid * MERGE_SUPERACCS_SIZE + i) * BIN_COUNT + lid];

        d_PartialSuperaccs[gid * BIN_COUNT + lid] = sum;
    }

    __syncthreads();
    if (lid == 0) {
        int imin = 0;
        int imax = 38;
        Normalize(&d_PartialSuperaccs[gid * BIN_COUNT], &imin, &imax);
    }

    __syncthreads();
    if ((lid < BIN_COUNT) && (gid == 0)) {
        long sum = 0;

        for(unsigned int i = 0; i < gridDim.x * blockDim.x / blockDim.x; i++)
            sum += d_PartialSuperaccs[i * BIN_COUNT + lid];

        d_PartialSuperaccs[lid] = sum;

        __syncthreads();
        if (lid == 0)
            d_Res[0] = Round(d_PartialSuperaccs);
    }
}

extern "C" void launch_ExDOT(
    long *d_PartialSuperaccs,
    double *d_a,
    double *d_b,
    const unsigned int NbElements
){

    ExDOT<<< PARTIAL_SUPERACCS_COUNT,WORKGROUP_SIZE>>>(d_PartialSuperaccs, d_a, d_b, NbElements);
}

extern "C" void launch_ExDOTComplete(
    double *d_Res,
    long *d_PartialSuperaccs,
    unsigned int PartialSuperaccusCount
){
    int GridSize =  PARTIAL_SUPERACCS_COUNT/MERGE_SUPERACCS_SIZE;
    ExDOTComplete<<<GridSize, MERGE_WORKGROUP_SIZE>>>(d_Res, d_PartialSuperaccs, PartialSuperaccusCount);
}

