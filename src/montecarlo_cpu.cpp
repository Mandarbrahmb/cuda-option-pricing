#include <cmath>
#include <chrono>
#include <iostream>
#include <random>
#include <vector>

double monteCarloCallCPU(
    double S0,
    double K,
    double r,
    double sigma,
    double T,
    int numPaths,
    int numSteps
) {
    std::mt19937 gen(42);
    std::normal_distribution<double> dist(0.0, 1.0);

    double dt = T / numSteps;
    double drift = (r - 0.5 * sigma * sigma) * dt;
    double diffusion = sigma * std::sqrt(dt);

    double payoffSum = 0.0;

    for (int path = 0; path < numPaths; path++) {
        double S = S0;

        for (int step = 0; step < numSteps; step++) {
            double z = dist(gen);
            S = S * std::exp(drift + diffusion * z);
        }

        double payoff = std::max(S - K, 0.0);
        payoffSum += payoff;
    }

    return std::exp(-r * T) * (payoffSum / numPaths);
}

int main() {
    double S0 = 100.0;
    double K = 100.0;
    double r = 0.05;
    double sigma = 0.2;
    double T = 1.0;

    int numPaths = 100000;
    int numSteps = 252;

    auto start = std::chrono::high_resolution_clock::now();

    double price = monteCarloCallCPU(S0, K, r, sigma, T, numPaths, numSteps);

    auto end = std::chrono::high_resolution_clock::now();
    double elapsedMs =
        std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << "CPU Monte Carlo European Call Option Price: " << price << "\n";
    std::cout << "CPU Runtime: " << elapsedMs << " ms\n";

    return 0;
}
