#include <cstdio>
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include "kmeans.h"
#include "kmeans_utils.h"
#include <cuda_profiler_api.h>

// CUDA kernel to normalize the centroids by the count of points assigned to each
__global__ void normalizeCentroids(double *centroids, const int *counts, int k, int dims)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Normalize each centroid's values by dividing by the number of points assigned to it
    if (idx < k * dims)
    {
        int clusterId = idx / dims;
        if (counts[clusterId] > 0)
        {
            centroids[idx] /= counts[clusterId]; // Prevent division by zero
        }
    }
}

// CUDA kernel to assign points to the nearest centroid using shared memory
__global__ void assignPointsToCentroidsShmem(const double *data, double *centroids, int *labels, int numPoints, int k, int dims)
{
    extern __shared__ double sharedCentroids[]; // Allocate shared memory for centroids
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Load centroids from global memory to shared memory
    for (int d = tid; d < k * dims; d += blockDim.x)
    {
        sharedCentroids[d] = centroids[d];
    }
    __syncthreads(); // Ensure all threads have loaded centroids

    if (idx < numPoints)
    {
        double minDist = 1e20; // Initialize the minimum distance with a large value
        int closestCentroid = -1;

        // Find the closest centroid by calculating the Euclidean distance
        for (int j = 0; j < k; ++j)
        {
            double dist = 0.0;
            for (int d = 0; d < dims; ++d)
            {
                double diff = data[idx * dims + d] - sharedCentroids[j * dims + d];
                dist += diff * diff;
            }
            if (dist < minDist)
            {
                minDist = dist;
                closestCentroid = j; // Update the closest centroid
            }
        }
        labels[idx] = closestCentroid; // Assign the point to the closest centroid
    }
}

// CUDA kernel to assign points to the nearest centroid using basic global memory (no shared memory)
__global__ void assignPointsToCentroidsBasic(const double *data, double *centroids, int *labels, int numPoints, int k, int dims)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPoints)
    {
        double minDist = 1e20;
        int closestCentroid = -1;

        // Same process as the shared memory kernel, but working directly from global memory
        for (int j = 0; j < k; ++j)
        {
            double dist = 0.0;
            for (int d = 0; d < dims; ++d)
            {
                double diff = data[idx * dims + d] - centroids[j * dims + d];
                dist += diff * diff;
            }
            if (dist < minDist)
            {
                minDist = dist;
                closestCentroid = j;
            }
        }
        labels[idx] = closestCentroid; // Assign point to the closest centroid
    }
}

// Kernel to update centroids using shared memory
__global__ void updateCentroidsShmem(const double *data, double *centroids, const int *labels, int *counts, int numPoints, int k, int dims)
{
    extern __shared__ double sharedCentroids[]; // Shared memory for centroid updates
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // Zero out shared memory for centroids
    for (int d = tid; d < k * dims; d += blockDim.x)
    {
        sharedCentroids[d] = 0.0;
    }
    __syncthreads();

    // Accumulate the sum of points assigned to each centroid in shared memory
    if (idx < numPoints)
    {
        int clusterId = labels[idx];
        for (int d = 0; d < dims; ++d)
        {
            atomicAddDouble(&sharedCentroids[clusterId * dims + d], data[idx * dims + d]);
        }
        atomicAdd(&counts[clusterId], 1); // Update point counts for each cluster
    }
    __syncthreads();

    // Sum reduction: accumulate shared centroids back to global memory
    for (int d = tid; d < k * dims; d += blockDim.x)
    {
        atomicAddDouble(&centroids[d], sharedCentroids[d]);
    }
}

// Kernel to update centroids using global memory (basic version)
__global__ void updateCentroidsBasic(const double *data, double *centroids, const int *labels, int *counts, int numPoints, int k, int dims)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numPoints)
    {
        int clusterId = labels[idx];

        // Use atomicAdd to safely update counts and centroid coordinates
        atomicAdd(&counts[clusterId], 1);

        for (int d = 0; d < dims; ++d)
        {
            atomicAddDouble(&centroids[clusterId * dims + d], data[idx * dims + d]);
        }
    }
}

// Wrapper function to manage CUDA kernel execution and memory transfers
void KMeansCudaWrapper(int k, int dims, int maxNumIter, double threshold, int seed, const std::vector<std::vector<double>> &data, bool outputCentroids, bool useSharedMemory)
{
    cudaProfilerStart(); // Start CUDA profiler for performance analysis
    int numPoints = data.size();

    // Host memory allocation
    double *h_data = new double[numPoints * dims];
    double *h_centroids = new double[k * dims];
    double *h_oldCentroids = new double[k * dims];
    int *h_labels = new int[numPoints];
    int *h_counts = new int[k];

    // Copy data from std::vector to host memory
    for (int i = 0; i < numPoints; ++i)
    {
        for (int d = 0; d < dims; ++d)
        {
            h_data[i * dims + d] = data[i][d];
        }
    }

    // Device memory allocation
    double *d_data, *d_centroids;
    int *d_labels, *d_counts;
    cudaMalloc(&d_data, numPoints * dims * sizeof(double));
    cudaMalloc(&d_centroids, k * dims * sizeof(double));
    cudaMalloc(&d_labels, numPoints * sizeof(int));
    cudaMalloc(&d_counts, k * sizeof(int));

    // Copy data from host to device memory
    cudaMemcpy(d_data, h_data, numPoints * dims * sizeof(double), cudaMemcpyHostToDevice);

    // Randomly initialize centroids by selecting k random points from the data
    kmeans_srand(seed);
    for (int i = 0; i < k; ++i)
    {
        int index = kmeans_rand() % numPoints;
        for (int d = 0; d < dims; ++d)
        {
            h_centroids[i * dims + d] = data[index][d];
        }
    }
    cudaMemcpy(d_centroids, h_centroids, k * dims * sizeof(double), cudaMemcpyHostToDevice);

    // Kernel configuration
    int threadsPerBlock = 256;
    int numBlocks = (numPoints + threadsPerBlock - 1) / threadsPerBlock;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    bool converged = false;
    for (int iter = 0; iter < maxNumIter && !converged; ++iter)
    {
        // Reset point counts on the device
        cudaMemset(d_counts, 0, k * sizeof(int));

        // Assign points to centroids using either shared memory or basic kernel
        if (useSharedMemory)
        {
            assignPointsToCentroidsShmem<<<numBlocks, threadsPerBlock, k * dims * sizeof(double)>>>(d_data, d_centroids, d_labels, numPoints, k, dims);
        }
        else
        {
            assignPointsToCentroidsBasic<<<numBlocks, threadsPerBlock>>>(d_data, d_centroids, d_labels, numPoints, k, dims);
        }
        cudaDeviceSynchronize();

        // Store old centroids for convergence checking
        cudaMemcpy(h_oldCentroids, d_centroids, k * dims * sizeof(double), cudaMemcpyDeviceToHost);

        // Update centroids using either shared memory or basic kernel
        if (useSharedMemory)
        {
            updateCentroidsShmem<<<numBlocks, threadsPerBlock, k * dims * sizeof(double)>>>(d_data, d_centroids, d_labels, d_counts, numPoints, k, dims);
        }
        else
        {
            updateCentroidsBasic<<<numBlocks, threadsPerBlock>>>(d_data, d_centroids, d_labels, d_counts, numPoints, k, dims);
        }
        cudaDeviceSynchronize();

        // Normalize centroids based on the counts
        normalizeCentroids<<<(k * dims + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock>>>(d_centroids, d_counts, k, dims);
        cudaDeviceSynchronize();

        // Copy centroids back to host memory for convergence check
        cudaMemcpy(h_centroids, d_centroids, k * dims * sizeof(double), cudaMemcpyDeviceToHost);

        // Check for convergence by comparing current and old centroids
        converged = true;
        for (int j = 0; j < k; ++j)
        {
            for (int d = 0; d < dims; ++d)
            {
                double diff = h_centroids[j * dims + d] - h_oldCentroids[j * dims + d];
                if (fabs(diff) > threshold * 10) // Larger threshold for floating-point inaccuracy
                {
                    converged = false;
                    break;
                }
            }
            if (!converged)
            {
                break;
            }
        }
    }

    // Stop timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    float timePerIteration = milliseconds / maxNumIter;

    // Print the total elapsed time and time per iteration
    std::cout << "Total elapsed time: " << std::fixed << std::setprecision(6) << milliseconds << " ms" << std::endl;
    std::cout << "Average time per iteration: " << std::fixed << std::setprecision(6) << timePerIteration << " ms" << std::endl;

    // Copy labels and centroids back to host for output
    cudaMemcpy(h_labels, d_labels, numPoints * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_centroids, d_centroids, k * dims * sizeof(double), cudaMemcpyDeviceToHost);

    // Output final centroids or labels
    if (outputCentroids)
    {
        for (int clusterId = 0; clusterId < k; ++clusterId)
        {
            std::cout << clusterId << " ";
            for (int d = 0; d < dims; ++d)
            {
                std::cout << std::fixed << std::setprecision(6) << h_centroids[clusterId * dims + d] << " ";
            }
            std::cout << std::endl;
        }
    }
    else
    {
        std::cout << "clusters:";
        for (int i = 0; i < numPoints; ++i)
        {
            std::cout << " " << h_labels[i];
        }
        std::cout << std::endl;
    }

    cudaProfilerStop(); // Stop CUDA profiler

    // Free device memory
    cudaFree(d_data);
    cudaFree(d_centroids);
    cudaFree(d_labels);
    cudaFree(d_counts);

    // Free host memory
    delete[] h_data;
    delete[] h_centroids;
    delete[] h_oldCentroids;
    delete[] h_labels;
    delete[] h_counts;

    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
