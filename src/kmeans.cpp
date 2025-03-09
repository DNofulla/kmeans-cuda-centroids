#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include "kmeans.h"

// Function to read dataset from input file
std::vector<std::vector<double>> readInputFile(const std::string &filename, int &numPoints, int dims)
{
    std::ifstream inputFile(filename);
    if (!inputFile)
    {
        std::cerr << "Error opening input file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    std::vector<std::vector<double>> data;
    std::string line;
    numPoints = 0;

    while (std::getline(inputFile, line))
    {
        std::istringstream ss(line);
        std::vector<double> point(dims);
        for (int i = 0; i < dims; ++i)
        {
            ss >> point[i];
        }
        data.push_back(point);
        numPoints++;
    }

    return data;
}

int main(int argc, char *argv[])
{
    // Command line argument variables
    int k = 0;
    int dims = 0;
    std::string inputFilename;
    int maxNumIter = 150;
    double threshold = 1e-5;
    bool outputCentroids = false;
    int seed = 8675309; // Default seed
    bool useCuda = false;
    bool useSharedMemory = false;
    bool useThrust = false;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "-k" && i + 1 < argc)
        {
            k = std::stoi(argv[++i]);
        }
        else if (arg == "-d" && i + 1 < argc)
        {
            dims = std::stoi(argv[++i]);
        }
        else if (arg == "-i" && i + 1 < argc)
        {
            inputFilename = argv[++i];
        }
        else if (arg == "-m" && i + 1 < argc)
        {
            maxNumIter = std::stoi(argv[++i]);
        }
        else if (arg == "-t" && i + 1 < argc)
        {
            threshold = std::stod(argv[++i]);
        }
        else if (arg == "-c")
        {
            outputCentroids = true;
        }
        else if (arg == "-s" && i + 1 < argc)
        {
            seed = std::stoi(argv[++i]);
        }
        else if (arg == "-cuda")
        {
            useCuda = true;
        }
        else if (arg == "-shmem")
        {
            useSharedMemory = true;
        }
        else if (arg == "-thrust")
        {
            useThrust = true;
        }
        else
        {
            std::cerr << "Unknown or incomplete argument: " << arg << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    // Check for required arguments
    if (k <= 0 || dims <= 0 || inputFilename.empty())
    {
        std::cerr << "Usage: " << argv[0] << " -k num_cluster -d dims -i inputfilename [-m max_num_iter] [-t threshold] [-c] [-s seed] [-cuda] [-shmem]" << std::endl;
        return EXIT_FAILURE;
    }

    // Set random seed
    kmeans_srand(seed);

    // Read dataset from input file
    int numPoints;
    std::vector<std::vector<double>> data = readInputFile(inputFilename, numPoints, dims);

    // Call either CPU, CUDA or Thrust implementation based on flag
    if (useThrust)
    {
#ifdef USE_CUDA
        std::cout << "Running Thrust implementation..." << std::endl;
        KMeansThrust(k, dims, maxNumIter, threshold, seed, data, outputCentroids);
#else
        std::cerr << "Thrust is only available when compiled with CUDA support." << std::endl;
#endif
    }
    else if (useCuda)
    {
#ifdef USE_CUDA
        std::cout << "Running CUDA implementation..." << (useSharedMemory ? " with Shared Memory" : " Basic") << std::endl;
        KMeansCudaWrapper(k, dims, maxNumIter, threshold, seed, data, outputCentroids, useSharedMemory);
#else
        std::cerr << "CUDA support is not enabled for this build." << std::endl;
#endif
    }
    else
    {
        std::cout << "Running CPU implementation..." << std::endl;
        KMeansSequential(k, dims, maxNumIter, threshold, seed, data, outputCentroids);
    }

    return 0;
}