#include <cmath>
#include <iostream>
#include <string>
#include <stdexcept>
#include <curand_kernel.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__      \
                      << " -> " << cudaGetErrorString(err) << std::endl;      \
            return 1;                                                         \
        }                                                                     \
    } while (0)

enum OptionType {
    EUROPEAN = 0,
    ASIAN = 1
};

OptionType parseOptionType(const std::string& optionTypeStr) {
    if (optionTypeStr == "european") return EUROPEAN;
    if (optionTypeStr == "asian") return ASIAN;
    throw std::invalid_argument("Invalid option type");
}

__global__ void setupCurand(curandState *states, unsigned long seed, int numPaths) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPaths) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

__global__ void monteCarloKernel(
    curandState *states,
    float *payoffs,
    int optionType,
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
    float runningSum = 0.0f;

    for (int step = 0; step < numSteps; step++) {
        float z = curand_normal(&localState);
        S *= expf(drift + diffusion * z);

        if (optionType == ASIAN) {
            runningSum += S;
        }
    }

    float payoff = 0.0f;
    if (optionType == EUROPEAN) {
        payoff = fmaxf(S - K, 0.0f);
    } else {
        float averagePrice = runningSum / numSteps;
        payoff = fmaxf(averagePrice - K, 0.0f);
    }

    payoffs[idx] = payoff;
    states[idx] = localState;
}

__global__ void reduceSum(const float *input, float *output, int n) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    float sum = 0.0f;
    if (i < n) sum += input[i];
    if (i + blockDim.x < n) sum += input[i + blockDim.x];

    sdata[tid] = sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

void printUsage(const char* programName) {
    std::cout << "Usage:\n";
    std::cout << "  " << programName << " <optionType> <numPaths> <numSteps>\n";
    std::cout << "  " << programName << " <optionType> <numPaths> <numSteps> <S0> <K> <r> <sigma> <T>\n";
    std::cout << "\noptionType must be: european or asian\n";
    std::cout << "\nExamples:\n";
    std::cout << "  " << programName << " european 100000 252\n";
    std::cout << "  " << programName << " asian 100000 252\n";
    std::cout << "  " << programName << " european 100000 252 100 100 0.05 0.2 1.0\n";
}

int main(int argc, char* argv[]) {
    OptionType optionType = EUROPEAN;
    std::string optionTypeStr = "european";

    float S0 = 100.0f;
    float K = 100.0f;
    float r = 0.05f;
    float sigma = 0.2f;
    float T = 1.0f;

    int numPaths = 100000;
    int numSteps = 252;

    try {
        if (argc == 4) {
            optionTypeStr = argv[1];
            optionType = parseOptionType(optionTypeStr);
            numPaths = std::stoi(argv[2]);
            numSteps = std::stoi(argv[3]);
        } else if (argc == 9) {
            optionTypeStr = argv[1];
            optionType = parseOptionType(optionTypeStr);
            numPaths = std::stoi(argv[2]);
            numSteps = std::stoi(argv[3]);
            S0 = std::stof(argv[4]);
            K = std::stof(argv[5]);
            r = std::stof(argv[6]);
            sigma = std::stof(argv[7]);
            T = std::stof(argv[8]);
        } else if (argc != 1) {
            printUsage(argv[0]);
            return 1;
        }
    } catch (...) {
        std::cerr << "Error: invalid command-line arguments.\n";
        printUsage(argv[0]);
        return 1;
    }

    if (numPaths <= 0 || numSteps <= 0 || S0 <= 0.0f || K <= 0.0f || sigma < 0.0f || T <= 0.0f) {
        std::cerr << "Error: arguments must satisfy:\n";
        std::cerr << "  numPaths > 0, numSteps > 0, S0 > 0, K > 0, sigma >= 0, T > 0\n";
        return 1;
    }

    int threadsPerBlock = 256;
    int simBlocks = (numPaths + threadsPerBlock - 1) / threadsPerBlock;

    float *d_payoffs = nullptr;
    curandState *d_states = nullptr;

    CUDA_CHECK(cudaMalloc(&d_payoffs, numPaths * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_states, numPaths * sizeof(curandState)));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    setupCurand<<<simBlocks, threadsPerBlock>>>(d_states, 42, numPaths);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    monteCarloKernel<<<simBlocks, threadsPerBlock>>>(
        d_states, d_payoffs, static_cast<int>(optionType), S0, K, r, sigma, T, numSteps, numPaths
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    int currentSize = numPaths;
    float *d_input = d_payoffs;
    float *d_output = nullptr;

    while (currentSize > 1) {
        int reduceBlocks = (currentSize + (threadsPerBlock * 2 - 1)) / (threadsPerBlock * 2);
        CUDA_CHECK(cudaMalloc(&d_output, reduceBlocks * sizeof(float)));

        reduceSum<<<reduceBlocks, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(
            d_input, d_output, currentSize
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        if (d_input != d_payoffs) {
            CUDA_CHECK(cudaFree(d_input));
        }

        d_input = d_output;
        d_output = nullptr;
        currentSize = reduceBlocks;
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsedMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedMs, start, stop));

    float payoffSum = 0.0f;
    CUDA_CHECK(cudaMemcpy(&payoffSum, d_input, sizeof(float), cudaMemcpyDeviceToHost));

    double price = std::exp(-r * T) * (static_cast<double>(payoffSum) / numPaths);

    std::cout << "GPU Monte Carlo " << optionTypeStr << " Call Option Price: " << price << "\n";
    std::cout << "GPU Total Runtime: " << elapsedMs << " ms\n";
    std::cout << "Parameters: optionType=" << optionTypeStr
              << ", numPaths=" << numPaths
              << ", numSteps=" << numSteps
              << ", S0=" << S0
              << ", K=" << K
              << ", r=" << r
              << ", sigma=" << sigma
              << ", T=" << T << "\n";

    if (d_input != d_payoffs) {
        CUDA_CHECK(cudaFree(d_input));
    }
    CUDA_CHECK(cudaFree(d_payoffs));
    CUDA_CHECK(cudaFree(d_states));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}
