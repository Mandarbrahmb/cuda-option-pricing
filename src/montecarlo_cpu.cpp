#include <cmath>
#include <chrono>
#include <iostream>
#include <random>
#include <string>
#include <algorithm>

enum class OptionType {
    EUROPEAN,
    ASIAN
};

OptionType parseOptionType(const std::string& optionTypeStr) {
    if (optionTypeStr == "european") return OptionType::EUROPEAN;
    if (optionTypeStr == "asian") return OptionType::ASIAN;
    throw std::invalid_argument("Invalid option type");
}

double monteCarloCallCPU(
    OptionType optionType,
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
        double runningSum = 0.0;

        for (int step = 0; step < numSteps; step++) {
            double z = dist(gen);
            S *= std::exp(drift + diffusion * z);

            if (optionType == OptionType::ASIAN) {
                runningSum += S;
            }
        }

        double payoff = 0.0;
        if (optionType == OptionType::EUROPEAN) {
            payoff = std::max(S - K, 0.0);
        } else {
            double averagePrice = runningSum / numSteps;
            payoff = std::max(averagePrice - K, 0.0);
        }

        payoffSum += payoff;
    }

    return std::exp(-r * T) * (payoffSum / numPaths);
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
    OptionType optionType = OptionType::EUROPEAN;
    std::string optionTypeStr = "european";

    double S0 = 100.0;
    double K = 100.0;
    double r = 0.05;
    double sigma = 0.2;
    double T = 1.0;

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
            S0 = std::stod(argv[4]);
            K = std::stod(argv[5]);
            r = std::stod(argv[6]);
            sigma = std::stod(argv[7]);
            T = std::stod(argv[8]);
        } else if (argc != 1) {
            printUsage(argv[0]);
            return 1;
        }
    } catch (...) {
        std::cerr << "Error: invalid command-line arguments.\n";
        printUsage(argv[0]);
        return 1;
    }

    if (numPaths <= 0 || numSteps <= 0 || S0 <= 0.0 || K <= 0.0 || sigma < 0.0 || T <= 0.0) {
        std::cerr << "Error: arguments must satisfy:\n";
        std::cerr << "  numPaths > 0, numSteps > 0, S0 > 0, K > 0, sigma >= 0, T > 0\n";
        return 1;
    }

    auto start = std::chrono::high_resolution_clock::now();
    double price = monteCarloCallCPU(optionType, S0, K, r, sigma, T, numPaths, numSteps);
    auto end = std::chrono::high_resolution_clock::now();

    double elapsedMs =
        std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << "CPU Monte Carlo " << optionTypeStr << " Call Option Price: " << price << "\n";
    std::cout << "CPU Runtime: " << elapsedMs << " ms\n";
    std::cout << "Parameters: optionType=" << optionTypeStr
              << ", numPaths=" << numPaths
              << ", numSteps=" << numSteps
              << ", S0=" << S0
              << ", K=" << K
              << ", r=" << r
              << ", sigma=" << sigma
              << ", T=" << T << "\n";

    return 0;
}
