//
// Created by mrp on 12.03.25.
//
#include "Solver.hpp"
#include "dirichlet.h"

#include <random>
#include <vector>
#include <numeric>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <numeric>
/**
 * Generates a random vector following an n-dimensional Dirichlet distribution.
 * @param alpha Vector of concentration parameters (alpha_i > 0).
 * @param engine Random number engine (e.g., std::mt19937).
 * @return Vector following Dirichlet(alpha) distribution, where elements sum to 1.
 */
// Generates a random vector from a Dirichlet distribution
std::vector<double> generate_dirichlet(const std::vector<double>& alpha, std::mt19937& gen) {
    if (alpha.empty()) throw std::invalid_argument("Alpha vector cannot be empty.");
    for (double a : alpha) {
        if (a <= 0.0) throw std::invalid_argument("All alpha values must be positive.");
    }

    size_t n = alpha.size();
    std::vector<double> y(n);
    for (size_t i = 0; i < n; ++i) {
        std::gamma_distribution<double> gamma(alpha[i], 1.0);
        y[i] = gamma(gen);
    }

    double sum = std::accumulate(y.begin(), y.end(), 0.0);
    std::vector<double> x(n);
    for (size_t i = 0; i < n; ++i) {
        x[i] = y[i] / sum;
    }
    return x;
}

std::vector<double> generate_dirichlet(size_t n, double alpha, std::mt19937& gen) {
    std::vector<double> y(n);
    std::gamma_distribution<double> gamma(alpha, 1.0);
    for (size_t i = 0; i < n; ++i) {
        y[i] = gamma(gen);
    }

    double sum = std::accumulate(y.begin(), y.end(), 0.0);
    std::vector<double> x(n);
    for (size_t i = 0; i < n; ++i) {
        x[i] = y[i] / sum;
    }
    return x;
}


// Generates a sample from a Gumbel distribution
double generate_gumbel(double mu, double beta, std::mt19937& gen) {
    if (beta <= 0.0) throw std::invalid_argument("Beta must be positive.");

    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    double u = uniform(gen);
    // Inverse CDF: mu - beta * ln(-ln(U))
    return mu - beta * std::log(-std::log(u));
}


// Tests the Dirichlet generator
void test_dirichlet(const std::vector<double>& alpha, size_t num_samples = 10000) {
    std::random_device rd;
    std::mt19937 gen(rd());

    size_t n = alpha.size();
    double alpha_sum = std::accumulate(alpha.begin(), alpha.end(), 0.0);

    // Compute theoretical means
    std::vector<double> theoretical_means(n);
    for (size_t i = 0; i < n; ++i) {
        theoretical_means[i] = alpha[i] / alpha_sum;
    }

    // Store samples
    std::vector<std::vector<double>> samples(num_samples, std::vector<double>(n));

    // Generate samples and perform checks
    for (size_t s = 0; s < num_samples; ++s) {
        samples[s] = generate_dirichlet(alpha, gen);
        double sum = std::accumulate(samples[s].begin(), samples[s].end(), 0.0);

        // Check sum is approximately 1
        if (std::abs(sum - 1.0) > 1e-10) {
            std::cerr << "Sample " << s << " does not sum to 1: sum = " << sum << std::endl;
        }

        // Check range
        for (double x : samples[s]) {
            if (x < 0.0 || x > 1.0) {
                std::cerr << "Component out of range: " << x << std::endl;
            }
        }
    }

    // Compute sample means
    std::vector<double> sample_means(n, 0.0);
    for (size_t i = 0; i < n; ++i) {
        for (size_t s = 0; s < num_samples; ++s) {
            sample_means[i] += samples[s][i];
        }
        sample_means[i] /= num_samples;
    }

    // Compare sample means to theoretical means
    for (size_t i = 0; i < n; ++i) {
        double diff = std::abs(sample_means[i] - theoretical_means[i]);
        if (diff > 1e-2) {
            std::cerr << "Mean for component " << i << " does not match: sample = " << sample_means[i]
                      << ", theoretical = " << theoretical_means[i] << std::endl;
        }
    }

    // Success message
    std::cout << "All tests passed for alpha = [";
    for (double a : alpha) std::cout << a << " ";
    std::cout << "]" << std::endl;
}


// Tests the Gumbel generator
void test_gumbel(double mu, double beta, size_t num_samples = 100000) {
    std::random_device rd;
    std::mt19937 gen(rd());

    // Theoretical mean and variance
    const double euler_gamma = 0.5772156649015329;
    double theoretical_mean = mu + beta * euler_gamma;
    double theoretical_variance = (M_PI * M_PI * beta * beta) / 6.0;

    // Generate samples
    std::vector<double> samples(num_samples);
    for (size_t i = 0; i < num_samples; ++i) {
        samples[i] = generate_gumbel(mu, beta, gen);
    }

    // Compute sample mean
    double sample_mean = std::accumulate(samples.begin(), samples.end(), 0.0) / num_samples;

    // Compute sample variance
    double sample_variance = 0.0;
    for (double x : samples) {
        double diff = x - sample_mean;
        sample_variance += diff * diff;
    }
    sample_variance /= (num_samples - 1);

    // Check mean and variance
    if (std::abs(sample_mean - theoretical_mean) > 0.05) {
        std::cerr << "Gumbel mean mismatch: sample = " << sample_mean
                  << ", theoretical = " << theoretical_mean << std::endl;
    }
    if (std::abs(sample_variance - theoretical_variance) > 0.1) {
        std::cerr << "Gumbel variance mismatch: sample = " << sample_variance
                  << ", theoretical = " << theoretical_variance << std::endl;
    }

    std::cout << "Gumbel tests passed for mu = " << mu << ", beta = " << beta << std::endl;
}



std::vector<double> policy_from_scores(const std::vector<int>& scores, int moves_played, double discount) {
    // Step 1: Calculate relative_end
    int relative_end = 22 - (moves_played / 2);

    // Step 2: Find the maximum score
    double best = *std::max_element(scores.begin(), scores.end());

    // Step 3: Initialize policy as a copy of scores
    std::vector<double> policy(scores.begin(), scores.end());

    // Step 4: Apply transformations based on best
    if (best > 0) {
        for (size_t i = 0; i < policy.size(); ++i) {
            if (policy[i] < 0) {
                policy[i] = 0;
            } else if (policy[i] > 0) {
                policy[i] = std::pow(discount, relative_end - policy[i]);
            }
        }
    } else if (best == 0) {
        for (size_t i = 0; i < policy.size(); ++i) {
            if (policy[i] < 0) {
                policy[i] = 0;
            } else if (policy[i] == 0) {
                policy[i] = 1;
            }
        }
    } else {
        // When best < 0, assume correction to apply transformation to all elements
        for (size_t i = 0; i < policy.size(); ++i) {
            if (policy[i] > -100) // can move
                policy[i] = std::pow(discount, relative_end - policy[i]);
            else policy[i] = 0;
        }
    }

    // Step 5: Normalize policy
    double sum_policy = std::accumulate(policy.begin(), policy.end(), 0.0);
    for (size_t i = 0; i < policy.size(); ++i) {
        policy[i] /= sum_policy;
    }

    return policy;
}

using namespace GameSolver::Connect4;

int main() {
    constexpr size_t rounds = 100000;
    std::random_device rd;
    std::mt19937 gen(rd());
    const double p = 0.2;
    const double q = 1-p;
    const int max_moves = 7*6;

    Solver solver;
    const double discount = 0.9;
    const bool weak = false;

    std::string opening_book = "7x6.book";

    solver.loadBook(opening_book);

    for (size_t i = 0; i < rounds; ++i) {
        std::vector<char> moves;
        Position P; //starting position
        while(true) {
            for(char move : moves) {
                std::cout << move;
            }

            std::vector<int> scores = solver.analyze(P, weak);
            for (int score : scores) {
                std::cout << " " << score;
            }

            std::cout << std::endl;

            std::vector<double> policy = policy_from_scores(scores, moves.size(), discount);
            std::vector<double> noise = generate_dirichlet(7, 0.9, gen);
            for (size_t i = 0; i < policy.size(); ++i) {
                if (scores[i] < -100) noise[i] = 0;
                policy[i] = p*policy[i] + q*noise[i];
                policy[i] = generate_gumbel(0, 1, gen) +  log(policy[i]);
            }
            int next_move = std::distance(policy.begin(), std::max_element(policy.begin(), policy.end()));
            if((max_moves != moves.size() + 1) && P.canPlay(next_move)) {
                if(!P.isWinningMove(next_move))  {
                    moves.push_back('1' + next_move);
                    P.playCol(next_move);
                }
                else break;
            }
            else {
                break;
            }


        }
    }

}