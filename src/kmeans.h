#ifndef KMEANS_H
#define KMEANS_H

#include <vector>

// KMeans class declaration
// This class handles the K-Means clustering algorithm for different modes (CPU, CUDA, Thrust)
class KMeans
{
public:
    // Constructor to initialize KMeans with essential parameters
    KMeans(int k, int dims, int maxNumIter, double threshold, int seed);

    // Function to fit the KMeans model with the given data
    void fit(const std::vector<std::vector<double>> &data);

    // Function to print the final centroids (cluster centers)
    void printCentroids() const;

    // Function to print the labels assigned to each point
    void printLabels() const;

private:
    int k;            // Number of clusters (centroids)
    int dims;         // Dimensionality of the data points
    int maxNumIter;   // Maximum number of iterations for the algorithm
    double threshold; // Threshold for convergence (stopping condition)
    int seed;         // Seed for random number generation (for reproducibility)

    bool outputCentroids; // Flag to determine if the centroids should be printed
    bool useCuda;         // Flag to determine if CUDA implementation should be used
    bool useSharedMemory; // Flag to determine if shared memory in CUDA should be used
    bool useThrust;       // Flag to determine if Thrust library should be used

    std::vector<std::vector<double>> centroids; // Vector storing the centroids of clusters
    std::vector<int> labels;                    // Vector storing the labels for each data point
};

// Function declarations for CUDA and Thrust implementations
#ifdef USE_CUDA
// Wrapper function for the CUDA-based KMeans implementation
void KMeansCudaWrapper(int k, int dims, int maxNumIter, double threshold, int seed,
                       const std::vector<std::vector<double>> &data, bool outputCentroids, bool useSharedMemory);

// Wrapper function for the Thrust-based KMeans implementation
void KMeansThrust(int k, int dims, int maxNumIter, double threshold, int seed,
                  const std::vector<std::vector<double>> &data, bool outputCentroids);
#endif

// Function declaration for the CPU-based sequential KMeans implementation
void KMeansSequential(int k, int dims, int maxNumIter, double threshold, int seed,
                      const std::vector<std::vector<double>> &data, bool outputCentroids);

// Static variables and functions for random number generation
static unsigned long int next = 1;        // Seed for the random number generator
static unsigned long kmeans_rmax = 32767; // Maximum value for random number generation

// Function to generate a random number
static inline int kmeans_rand()
{
    next = next * 1103515245 + 12345;
    return (unsigned int)(next / 65536) % (kmeans_rmax + 1); // Generate a random number
}

// Function to set the seed for the random number generator
static inline void kmeans_srand(unsigned int seed)
{
    next = seed; // Set the seed for the random number generator
}

#endif // KMEANS_H