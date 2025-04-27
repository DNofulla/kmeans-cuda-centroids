# K-Means Clustering: CPU vs CUDA vs Thrust Comparison

## Overview

This project implements the K-Means clustering algorithm in C++ and CUDA C++, providing multiple execution modes to compare performance:

1.  **Sequential CPU:** A standard C++ implementation.
2.  **CUDA Basic:** Parallel implementation using custom CUDA kernels with global memory access.
3.  **CUDA Shared Memory:** Optimized CUDA kernel implementation leveraging shared memory to reduce global memory latency.
4.  **CUDA Thrust:** High-level parallel implementation using the CUDA Thrust library with custom functors.

The goal is to demonstrate and benchmark different approaches to parallelizing K-Means on NVIDIA GPUs. The application takes multi-dimensional data points from a text file and clusters them into a specified number of groups (`k`).

## Features

* Implementation of the standard K-Means clustering algorithm.
* Multiple execution backends: Sequential CPU, CUDA Kernels (Basic & Shared Memory), CUDA Thrust.
* Command-line interface for easy configuration (k, dimensions, input file, iterations, etc.).
* Custom CUDA kernels for point assignment and centroid updates.
* Shared memory optimization in CUDA kernels for improved performance.
* Use of `atomicAdd` for safe parallel updates in CUDA kernels (including a custom `atomicAddDouble` utility).
* High-level parallel implementation using Thrust algorithms (`thrust::transform`, `thrust::for_each`) and custom functors.
* Performance timing (total elapsed time and average time per iteration) for each implementation.
* Python script (`run_all.py`) to automate compilation and benchmarking across different implementations and datasets.

## Dependencies

* **C++ Compiler:** A C++ compiler supporting C++11 or later (e.g., GCC, Clang, MSVC).
* **CUDA Toolkit:** Required for CUDA and Thrust implementations. Version supporting Compute Capability `sm_60` or higher is needed for the Shared Memory and Thrust builds as configured in `run_all.py`.
* **Python 3:** Required to run the `run_all.py` benchmarking script.
* **NVIDIA GPU:** Required to run the CUDA and Thrust implementations.

## Building the Project

The project does not use a complex build system like CMake. You can compile the different versions directly using `g++` (for CPU) and `nvcc` (for CUDA/Thrust) as shown in the `run_all.py` script or the `run_configurations` file.

**Example Compile Commands:**

*(Ensure source files are in a `src/` directory and output goes to a `bin/` directory, or adjust paths accordingly)*

1.  **CPU Version:**
    ```bash
    mkdir -p bin src
    # Add your .cpp/.h files to src/
    g++ -o bin/kmeans_cpu src/kmeans.cpp src/kmeans_cpu.cpp -I src/ -std=c++11 -O3
    ```

2.  **CUDA Basic Version:**
    ```bash
    # Assumes kmeans_thrust.cu and kmeans_cpu.cpp are also in src/
    nvcc -o bin/kmeans_cuda src/kmeans.cpp src/kmeans_kernel.cu src/kmeans_thrust.cu src/kmeans_cpu.cpp -I src/ -lcudart -DUSE_CUDA -std=c++11 -O3
    ```

3.  **CUDA Shared Memory Version:** (Requires GPU with Compute Capability 6.0+)
    ```bash
    nvcc -arch=sm_60 -o bin/kmeans_shmem src/kmeans.cpp src/kmeans_kernel.cu src/kmeans_thrust.cu src/kmeans_cpu.cpp -I src/ -lcudart -DUSE_CUDA -std=c++11 -O3
    ```

4.  **Thrust Version:** (Requires GPU with Compute Capability 6.0+)
    ```bash
    nvcc -arch=sm_60 -o bin/kmeans_thrust src/kmeans.cpp src/kmeans_thrust.cu src/kmeans_kernel.cu src/kmeans_cpu.cpp -I src/ -lcudart -DUSE_CUDA -std=c++11 -O3
    ```

**Automated Compilation (using Python script):**

The provided `run_all.py` script can compile all versions:
```bash
python run_all.py # This will execute the compile_all() function first
```

## Usage

Run the desired executable from the `bin/` directory, providing the required arguments:

```bash
./bin/<executable_name> -k <num_clusters> -d <dimensions> -i <input_file.txt> [options]
```

**Required Arguments:**

* `-k <num_clusters>`: Number of clusters (centroids) to find.
* `-d <dimensions>`: Number of dimensions (features) per data point.
* `-i <input_file.txt>`: Path to the input data file.

**Optional Arguments:**

* `-m <max_num_iter>`: Maximum number of iterations (default: 150).
* `-t <threshold>`: Convergence threshold (default: 1e-5).
* `-c`: Output the final centroids instead of cluster labels.
* `-s <seed>`: Seed for random initialization (default: 8675309).
* `-cuda`: (Required for `kmeans_cuda`, `kmeans_shmem`, `kmeans_thrust`) Use a CUDA implementation.
* `-shmem`: (Required for `kmeans_shmem`) Use the shared memory CUDA kernel version.
* `-thrust`: (Required for `kmeans_thrust`) Use the Thrust implementation.

**Examples:**

* **Run CPU version:**
    ```bash
    ./bin/kmeans_cpu -k 16 -d 32 -i input/random-n65536-d32-c16.txt -c
    ```
* **Run CUDA Shared Memory version:**
    ```bash
    ./bin/kmeans_shmem -k 16 -d 32 -i input/random-n65536-d32-c16.txt -cuda -shmem -c
    ```
* **Run Thrust version:**
    ```bash
    ./bin/kmeans_thrust -k 16 -d 32 -i input/random-n65536-d32-c16.txt -cuda -thrust -c
    ```

### Input Data Format

The input data must be a text file (`.txt`) where:
* Each line represents a data point.
* Values (features/dimensions) on each line are separated by **spaces**.
* The number of values on each line must match the `-d <dimensions>` argument.

**Example `input.txt` (for `-d 2`):**
```text
1.0 2.5
1.5 2.8
5.0 8.1
8.0 8.5
1.2 0.9
9.0 9.2
```

### Output

The program outputs the following to the standard output (console):

1.  Indication of which implementation is running (CPU, CUDA Basic, CUDA Shared Memory, Thrust).
2.  Total elapsed time in milliseconds.
3.  Average time per iteration in milliseconds.
4.  Either:
    * The final cluster labels assigned to each point (default).
    * The coordinates of the final `k` cluster centroids (if `-c` flag is used).

**Example Output (with `-c`):**
```
Running CUDA implementation... with Shared Memory
Total elapsed time: 150.123456 ms
Average time per iteration: 1.000823 ms
0 1.150000 2.400000
1 7.000000 8.600000
... (rest of the centroids)
```

## Implementation Details

* **Initialization:** Centroids are initialized by randomly selecting `k` points from the input dataset.
* **Assignment Step:** Each point is assigned to the nearest centroid based on Euclidean distance.
    * *CPU:* Sequential loop.
    * *CUDA Kernels:* Parallel kernel (`assignPointsToCentroidsBasic` or `assignPointsToCentroidsShmem`). Shared memory version caches centroids locally per block.
    * *Thrust:* `thrust::transform` with `AssignCentroidFunctor`.
* **Update Step:** New centroids are calculated as the mean of all points assigned to each cluster.
    * *CPU:* Sequential loops for summation and averaging.
    * *CUDA Kernels:* Parallel kernel (`updateCentroidsBasic` or `updateCentroidsShmem`) using `atomicAdd` (and `atomicAddDouble`) for summation, followed by a `normalizeCentroids` kernel. Shared memory version performs partial sums within blocks.
    * *Thrust:* `thrust::for_each` with `UpdateCentroidFunctor` (using `atomicAddDouble`) for summation, followed by `thrust::for_each` with `NormalizeCentroidsFunctor` for normalization.
* **Convergence:** Iteration stops when the maximum number of iterations is reached or when the change in centroid positions between iterations is below the specified threshold.
* **`atomicAddDouble`:** A utility function (`kmeans_utils.h`) is provided to perform atomic addition on `double` values using `atomicCAS`, as native `atomicAdd` for doubles is not universally supported across all GPU architectures.

## Benchmarking

The `run_all.py` script facilitates running all compiled implementations across predefined input files (`input/random-*.txt`).

```bash
python run_all.py
```
This script will first compile all versions (if `compile_all()` is called) and then execute each implementation (`cpu`, `cuda_basic`, `cuda_shmem`, `thrust`) for each input file specified in the script, printing the timing results to the console. This allows for easy comparison of their performance characteristics.

