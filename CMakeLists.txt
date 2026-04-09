cmake_minimum_required(VERSION 3.18)
project(MonteCarloOptionPricing LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

add_executable(montecarlo_cpu src/montecarlo_cpu.cpp)
add_executable(montecarlo_gpu src/montecarlo_gpu.cu)

find_package(CUDAToolkit REQUIRED)
target_link_libraries(montecarlo_gpu PRIVATE CUDA::curand)
