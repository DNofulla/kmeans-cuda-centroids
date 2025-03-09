#include "kmeans.h"
#include <limits>   // For std::numeric_limits
#include <chrono>   // For timing (std::chrono)
#include <iostream> // For console output (std::cout)
#include <iomanip>  // For formatting output (std::setprecision)

// Constructor: Initializes the KMeans object with parameters
KMeans::KMeans(int k, int dims, int maxNumIter, double threshold, int seed)
    : k(k), dims(dims), maxNumIter(maxNumIter), threshold(threshold), seed(seed)
{
    // Seed the random number generator for reproducibility
    kmeans_srand(seed);
}

// K-means clustering algorithm
void KMeans::fit(const std::vector<std::vector<double>> &data)
{
    int numPoints = data.size();
    centroids.resize(k, std::vector<double>(dims)); // Initialize centroids
    labels.resize(numPoints, -1);                   // Initialize labels with -1

    // Step 1: Randomly initialize centroids by selecting k points from the data
    for (int i = 0; i < k; ++i)
    {
        int index = kmeans_rand() % numPoints;
        centroids[i] = data[index];
    }

    // Step 2: Main K-means loop to assign points and update centroids
    int iterations = 0;
    bool converged = false;
    auto start_time = std::chrono::high_resolution_clock::now(); // Start timing

    while (iterations < maxNumIter && !converged)
    {
        converged = true;
        iterations++;

        // Assign each point to the nearest centroid
        for (int i = 0; i < numPoints; ++i)
        {
            double minDist = std::numeric_limits<double>::max(); // Track minimum distance
            int closestCentroid = -1;

            // Compute distance between point and each centroid
            for (int j = 0; j < k; ++j)
            {
                double dist = 0.0;
                for (int d = 0; d < dims; ++d)
                {
                    dist += (data[i][d] - centroids[j][d]) * (data[i][d] - centroids[j][d]);
                }
                if (dist < minDist)
                {
                    minDist = dist;
                    closestCentroid = j; // Find the closest centroid
                }
            }

            // If the label has changed, mark convergence as false
            if (labels[i] != closestCentroid)
            {
                converged = false;
                labels[i] = closestCentroid;
            }
        }

        // Update centroids by averaging the points in each cluster
        std::vector<std::vector<double>> newCentroids(k, std::vector<double>(dims, 0.0));
        std::vector<int> count(k, 0);

        for (int i = 0; i < numPoints; ++i)
        {
            int clusterId = labels[i];
            for (int d = 0; d < dims; ++d)
            {
                newCentroids[clusterId][d] += data[i][d];
            }
            count[clusterId]++;
        }

        // Normalize centroids by the number of points in each cluster
        for (int j = 0; j < k; ++j)
        {
            if (count[j] > 0)
            {
                for (int d = 0; d < dims; ++d)
                {
                    newCentroids[j][d] /= count[j];
                }
            }
        }

        centroids = newCentroids; // Update the centroids for the next iteration
    }

    auto end_time = std::chrono::high_resolution_clock::now(); // End timing
    double elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    double time_per_iter = elapsed_time / iterations;

    // Output elapsed time and average time per iteration
    std::cout << "Total elapsed time: " << elapsed_time << " ms" << std::endl;
    std::cout << "Average time per iteration: " << time_per_iter << " ms" << std::endl;
}

// Print the final centroids
void KMeans::printCentroids() const
{
    for (int clusterId = 0; clusterId < k; ++clusterId)
    {
        std::cout << clusterId << " ";
        for (int d = 0; d < dims; ++d)
        {
            std::cout << std::fixed << std::setprecision(6) << centroids[clusterId][d] << " ";
        }
        std::cout << std::endl;
    }
}

// Print the labels assigned to each point
void KMeans::printLabels() const
{
    std::cout << "clusters:";
    for (const auto &label : labels)
    {
        std::cout << " " << label;
    }
    std::cout << std::endl;
}

// Sequential implementation of K-means for CPU
void KMeansSequential(int k, int dims, int maxNumIter, double threshold, int seed, const std::vector<std::vector<double>> &data, bool outputCentroids)
{
    int numPoints = data.size();
    std::vector<std::vector<double>> centroids(k, std::vector<double>(dims)); // Centroids
    std::vector<int> labels(numPoints, -1);                                   // Labels

    // Initialize centroids randomly
    kmeans_srand(seed);
    for (int i = 0; i < k; ++i)
    {
        int index = kmeans_rand() % numPoints;
        centroids[i] = data[index];
    }

    bool converged = false;
    int iterations = 0;

    // Start timing
    auto start = std::chrono::high_resolution_clock::now();

    while (iterations < maxNumIter && !converged)
    {
        iterations++;
        converged = true;

        // Assign points to nearest centroids
        for (int i = 0; i < numPoints; ++i)
        {
            double minDist = std::numeric_limits<double>::max();
            int closestCentroid = -1;

            // Calculate distance to each centroid
            for (int j = 0; j < k; ++j)
            {
                double dist = 0.0;
                for (int d = 0; d < dims; ++d)
                {
                    double diff = data[i][d] - centroids[j][d];
                    dist += diff * diff;
                }

                if (dist < minDist)
                {
                    minDist = dist;
                    closestCentroid = j;
                }
            }

            if (labels[i] != closestCentroid)
            {
                converged = false;
                labels[i] = closestCentroid;
            }
        }

        // Update centroids based on point assignments
        std::vector<std::vector<double>> newCentroids(k, std::vector<double>(dims, 0.0));
        std::vector<int> count(k, 0);

        for (int i = 0; i < numPoints; ++i)
        {
            int clusterId = labels[i];
            for (int d = 0; d < dims; ++d)
            {
                newCentroids[clusterId][d] += data[i][d];
            }
            count[clusterId]++;
        }

        // Normalize centroids
        for (int j = 0; j < k; ++j)
        {
            if (count[j] > 0)
            {
                for (int d = 0; d < dims; ++d)
                {
                    newCentroids[j][d] /= count[j];
                }
            }
        }

        centroids = newCentroids;
    }

    // Stop timing
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    double timePerIteration = elapsed.count() / iterations;

    // Output timing results
    std::cout << "Total elapsed time: " << elapsed.count() << " ms" << std::endl;
    std::cout << "Average time per iteration: " << timePerIteration << " ms" << std::endl;

    // Output results
    if (outputCentroids)
    {
        for (int clusterId = 0; clusterId < k; ++clusterId)
        {
            std::cout << clusterId << " ";
            for (int d = 0; d < dims; ++d)
            {
                std::cout << std::fixed << std::setprecision(6) << centroids[clusterId][d] << " ";
            }
            std::cout << std::endl;
        }
    }
    else
    {
        std::cout << "clusters:";
        for (int i = 0; i < numPoints; ++i)
        {
            std::cout << " " << labels[i];
        }
        std::cout << std::endl;
    }
}