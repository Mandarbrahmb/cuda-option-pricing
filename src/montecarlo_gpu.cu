#include <cmath>
#include <iostream>
#include <vector>
#include <chrono>
#include <curand_kernel.h>

__global__ void setupCurand(curandState *states, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, idx, 0, &states[idx]);
}

__global__ void monteCarloKernel(
    curandState *states,
    float *payoffs,
    float S0,
    float K,
    float r,
    float sigma,
    float T,
    int numSteps,
    int numPaths
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPaths) return;

    curandState localState = states[idx];

    float dt = T / numSteps;
    float drift = (r - 0.5f * sigma * sigma) * dt;
    float diffusion = sigma * sqrtf(dt);

    float S = S0;
    for (int step = 0; step < numSteps; step++) {
        float z = curand_normal(&localState);
        S = S * expf(drift + diffusion * z);
    }

    payoffs[idx] = fmaxf(S - K, 0.0f);
    states[idx] = localState;
}

int main() {
    float S0 = 100.0f;
    float K = 100.0f;
    float r = 0.05f;
    float sigma = 0.2f;
    float T = 1.0f;

    int numPaths = 100000;
    int numSteps = 252;

    float *d_payoffs = nullptr;
    curandState *d_states = nullptr;

    cudaMalloc(&d_payoffs, numPaths * sizeof(float));
    cudaMalloc(&d_states, numPaths * sizeof(curandState));

    int threadsPerBlock = 256;
    int blocks = (numPaths + threadsPerBlock - 1) / threadsPerBlock;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    setupCurand<<<blocks, threadsPerBlock>>>(d_states, 42);
    cudaDeviceSynchronize();

    monteCarloKernel<<<blocks, threadsPerBlock>>>(
        d_states, d_payoffs, S0, K, r, sigma, T, numSteps, numPaths
    );
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsedMs = 0.0f;
    cudaEventElapsedTime(&elapsedMs, start, stop);

    std::vector<float> h_payoffs(numPaths);
    cudaMemcpy(h_payoffs.data(), d_payoffs, numPaths * sizeof(float), cudaMemcpyDeviceToHost);

    double payoffSum = 0.0;
    for (int i = 0; i < numPaths; i++) {
        payoffSum += h_payoffs[i];
    }

    double price = std::exp(-r * T) * (payoffSum / numPaths);

    std::cout << "GPU Monte Carlo European Call Option Price: " << price << "\n";
    std::cout << "GPU Kernel Runtime: " << elapsedMs << " ms\n";

    cudaFree(d_payoffs);
    cudaFree(d_states);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
