#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/for_each.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/functional.h>
#include <thrust/pair.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <iomanip>
#include "kmeans.h"
#include "kmeans_utils.h"
#include <cuda_profiler_api.h>

// Functor to assign points to the nearest centroid
// This functor takes in the data and centroids, and for each point,
// it computes the Euclidean distance to every centroid and assigns
// the point to the closest centroid.
struct AssignCentroidFunctor
{
    // Pointers to the device data and centroids
    const double *d_data;
    const double *d_centroids;
    int dims, k;

    // Constructor to initialize the data members
    AssignCentroidFunctor(const double *_data, const double *_centroids, int _dims, int _k)
        : d_data(_data), d_centroids(_centroids), dims(_dims), k(_k) {}

    // Device-side operator to calculate the nearest centroid for each point
    __device__ int operator()(int idx) const
    {
        double minDist = 1e20;    // Track the minimum distance
        int closestCentroid = -1; // Closest centroid index

        // Loop through each centroid to compute the distance from the point
        for (int j = 0; j < k; ++j)
        {
            double dist = 0.0; // Initialize distance calculation
            // Compute squared Euclidean distance from point to centroid
            for (int d = 0; d < dims; ++d)
            {
                double diff = d_data[idx * dims + d] - d_centroids[j * dims + d];
                dist += diff * diff;
            }
            // If this centroid is closer, update the closest centroid
            if (dist < minDist)
            {
                minDist = dist;
                closestCentroid = j;
            }
        }
        return closestCentroid; // Return the index of the nearest centroid
    }
};

// Functor to update the centroids by summing the points assigned to each centroid
// It accumulates the points assigned to each centroid and calculates the sum for each dimension.
struct UpdateCentroidFunctor
{
    // Pointers to the data, labels, centroid sums, and counts
    const double *d_data;
    const int *d_labels;
    double *d_centroidSums;
    int *d_counts;
    int dims;

    // Constructor to initialize the data members
    UpdateCentroidFunctor(const double *_data, const int *_labels, double *_centroidSums, int *_counts, int _dims)
        : d_data(_data), d_labels(_labels), d_centroidSums(_centroidSums), d_counts(_counts), dims(_dims) {}

    // Device-side operator to sum up data points for each centroid
    __device__ void operator()(int idx)
    {
        // Get the centroid label for the current point
        int label = d_labels[idx];

        // Update centroid sums and counts atomically
        for (int d = 0; d < dims; ++d)
        {
            atomicAddDouble(&d_centroidSums[label * dims + d], d_data[idx * dims + d]);
        }
        atomicAdd(&d_counts[label], 1); // Atomically update the count for this centroid
    }
};

// Functor to normalize the centroids based on the number of points assigned to them
// It divides the centroid sum by the number of points to compute the mean for each dimension.
struct NormalizeCentroidsFunctor
{
    double *d_centroids;
    const int *d_counts;
    int dims;

    // Constructor to initialize centroids and counts
    NormalizeCentroidsFunctor(double *_centroids, const int *_counts, int _dims)
        : d_centroids(_centroids), d_counts(_counts), dims(_dims) {}

    // Device-side operator to normalize each centroid by dividing the sum by the count
    __device__ void operator()(int idx)
    {
        int centroidId = idx / dims; // Get the centroid ID from the index
        if (d_counts[centroidId] > 0)
        {
            d_centroids[idx] /= d_counts[centroidId]; // Normalize centroid values
        }
    }
};

// Thrust-based KMeans function
// This function uses Thrust to implement the KMeans algorithm, where the core computations (such as centroid assignment,
// centroid updating, and normalization) are performed using functors mapped over the data.
void KMeansThrust(int k, int dims, int maxNumIter, double threshold, int seed,
                  const std::vector<std::vector<double>> &data, bool outputCentroids)
{
    cudaProfilerStart();         // Start CUDA profiler to analyze kernel performance
    int numPoints = data.size(); // Get the number of points in the dataset

    // Host vectors to hold the data, centroids, and other information
    thrust::host_vector<double> h_data(numPoints * dims);
    thrust::host_vector<double> h_centroids(k * dims);    // Current centroids
    thrust::host_vector<double> h_oldCentroids(k * dims); // Centroids from the previous iteration
    thrust::host_vector<int> h_labels(numPoints);         // Cluster labels for each point
    thrust::host_vector<int> h_counts(k);                 // Number of points assigned to each centroid

    // Copy the input data to the host vector
    for (int i = 0; i < numPoints; ++i)
    {
        for (int d = 0; d < dims; ++d)
        {
            h_data[i * dims + d] = data[i][d]; // Flatten the 2D data array into a 1D vector
        }
    }

    // Device vectors to hold data, centroids, labels, and counts
    thrust::device_vector<double> d_data = h_data;
    thrust::device_vector<double> d_centroids(k * dims);
    thrust::device_vector<double> d_centroidSums(k * dims); // To store the sum of points for each centroid
    thrust::device_vector<int> d_labels(numPoints);         // To store the cluster labels
    thrust::device_vector<int> d_counts(k);                 // To store the number of points assigned to each centroid

    // Initialize centroids randomly by picking k points from the dataset
    kmeans_srand(seed);
    for (int i = 0; i < k; ++i)
    {
        int index = kmeans_rand() % numPoints; // Randomly select a point
        for (int d = 0; d < dims; ++d)
        {
            h_centroids[i * dims + d] = data[index][d]; // Assign this point as the centroid
        }
    }
    d_centroids = h_centroids; // Copy the initialized centroids to the device

    bool converged = false; // Flag to check if the algorithm has converged

    // Timing event start for performance measurement
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Main KMeans iteration loop
    for (int iter = 0; iter < maxNumIter && !converged; ++iter)
    {
        // Step 1: Assign each point to the nearest centroid using the AssignCentroidFunctor
        thrust::transform(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(numPoints),
                          d_labels.begin(), AssignCentroidFunctor(thrust::raw_pointer_cast(d_data.data()), thrust::raw_pointer_cast(d_centroids.data()), dims, k));

        // Zero out the sums and counts for the centroids
        thrust::fill(d_centroidSums.begin(), d_centroidSums.end(), 0.0); // Reset the centroid sums to zero
        thrust::fill(d_counts.begin(), d_counts.end(), 0);               // Reset the counts to zero

        // Step 2: Update the centroid sums and counts using the UpdateCentroidFunctor
        thrust::for_each(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(numPoints),
                         UpdateCentroidFunctor(thrust::raw_pointer_cast(d_data.data()),
                                               thrust::raw_pointer_cast(d_labels.data()),
                                               thrust::raw_pointer_cast(d_centroidSums.data()),
                                               thrust::raw_pointer_cast(d_counts.data()), dims));

        // Step 3: Copy the updated centroid sums back to the centroid array
        d_centroids = d_centroidSums;

        // Step 4: Normalize the centroids based on the number of points assigned to each one
        thrust::for_each(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(k * dims),
                         NormalizeCentroidsFunctor(thrust::raw_pointer_cast(d_centroids.data()),
                                                   thrust::raw_pointer_cast(d_counts.data()), dims));

        // Step 5: Check for convergence by comparing the current centroids with the centroids from the previous iteration
        thrust::copy(d_centroids.begin(), d_centroids.end(), h_centroids.begin());
        converged = true; // Assume convergence unless we find a significant difference

        // Compare each centroid dimension-by-dimension to detect changes
        for (int j = 0; j < k; ++j)
        {
            for (int d = 0; d < dims; ++d)
            {
                double diff = fabs(h_centroids[j * dims + d] - h_oldCentroids[j * dims + d]);
                if (diff > threshold)
                {
                    converged = false; // If any centroid moves by more than the threshold, the algorithm hasn't converged
                    break;
                }
            }
            if (!converged)
                break;
        }

        h_oldCentroids = h_centroids; // Update the old centroids for the next iteration
    }

    // Timing event stop for performance measurement
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    float timePerIteration = milliseconds / maxNumIter;

    // Print the total elapsed time and the average time per iteration
    std::cout << "Total elapsed time: " << std::fixed << std::setprecision(6) << milliseconds << " ms" << std::endl;
    std::cout << "Average time per iteration: " << std::fixed << std::setprecision(6) << timePerIteration << " ms" << std::endl;

    // If outputCentroids is true, print the final centroids
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
        // If outputCentroids is false, print the cluster labels for each point
        thrust::copy(d_labels.begin(), d_labels.end(), h_labels.begin());

        std::cout << "clusters:";
        for (int i = 0; i < numPoints; ++i)
        {
            std::cout << " " << h_labels[i];
        }
        std::cout << std::endl;
    }

    // Stop the CUDA profiler
    cudaProfilerStop();

    // Cleanup the CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
