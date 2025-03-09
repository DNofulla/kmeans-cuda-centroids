import subprocess
import os

# Input files
input_files = [
    "input/random-n2048-d16-c16.txt",
    "input/random-n16384-d24-c16.txt",
    "input/random-n65536-d32-c16.txt"
]

# Common KMeans parameters
k = 16
max_iter = 150
threshold = 1e-5
seed = 8675309

# Commands for compilation
compile_commands = [
    "g++ -o bin/kmeans_cpu src/kmeans.cpp src/kmeans_cpu.cpp -I src/",
    "nvcc -o bin/kmeans_cuda src/kmeans.cpp src/kmeans_kernel.cu src/kmeans_thrust.cu src/kmeans_cpu.cpp -I src/ -lcudart -DUSE_CUDA",
    "nvcc -arch=sm_60 -o bin/kmeans_shmem src/kmeans.cpp src/kmeans_kernel.cu src/kmeans_thrust.cu src/kmeans_cpu.cpp -I src/ -lcudart -DUSE_CUDA",
    "nvcc -arch=sm_60 -o bin/kmeans_thrust src/kmeans.cpp src/kmeans_thrust.cu src/kmeans_kernel.cu src/kmeans_cpu.cpp -I src/ -lcudart -DUSE_CUDA"
]

# Commands for running the implementations
run_commands = {
    "cpu": "./bin/kmeans_cpu -k {k} -d {dims} -i {input_file} -m {max_iter} -t {threshold} -s {seed} -c",
    "cuda_basic": "./bin/kmeans_cuda -k {k} -d {dims} -i {input_file} -m {max_iter} -t {threshold} -s {seed} -cuda -c",
    "cuda_shmem": "./bin/kmeans_shmem -k {k} -d {dims} -i {input_file} -m {max_iter} -t {threshold} -s {seed} -cuda -shmem -c",
    "thrust": "./bin/kmeans_thrust -k {k} -d {dims} -i {input_file} -m {max_iter} -t {threshold} -s {seed} -cuda -thrust -c"
}

# Dimensions for each input file
input_dims = {
    "input/random-n2048-d16-c16.txt": 16,
    "input/random-n16384-d24-c16.txt": 24,
    "input/random-n65536-d32-c16.txt": 32
}

def compile_all():
    for cmd in compile_commands:
        print(f"Compiling with command: {cmd}")
        subprocess.run(cmd, shell=True, check=True)

def run_all():
    for input_file in input_files:
        dims = input_dims[input_file]
        for implementation, run_cmd in run_commands.items():
            command = run_cmd.format(k=k, dims=dims, input_file=input_file, max_iter=max_iter, threshold=threshold, seed=seed)
            print(f"Running {implementation} with input {input_file}:")
            subprocess.run(command, shell=True, check=True)

if __name__ == "__main__":
    # Compile all implementations
    compile_all()
    
    # Run all implementations on each input file
    run_all()