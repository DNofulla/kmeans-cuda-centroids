#ifndef KMEANS_UTILS_H
#define KMEANS_UTILS_H

#include <cuda_runtime.h>

// Utility function for performing atomic addition on double precision floating-point numbers in CUDA.
// CUDA's built-in atomicAdd does not support double types directly on all architectures, so this function
// provides a workaround by using atomicCAS (atomic Compare And Swap) to achieve atomic addition for doubles.

static inline __device__ double atomicAddDouble(double *address, double val)
{
    // Cast the address of the double to an unsigned long long int (ULL) pointer
    unsigned long long int *address_as_ull = (unsigned long long int *)address;
    unsigned long long int old = *address_as_ull, assumed;

    // Use atomic Compare-And-Swap (atomicCAS) to safely add the double value atomically
    do
    {
        // Store the current value at the address
        assumed = old;

        // Calculate the new value and try to atomically swap it with the old one
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));

    } while (assumed != old); // Repeat if the value has changed since the last read

    // Return the final updated value as a double
    return __longlong_as_double(old);
}

#endif // KMEANS_UTILS_H