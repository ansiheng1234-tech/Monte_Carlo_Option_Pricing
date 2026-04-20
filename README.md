# Monte Carlo Methods for Options Pricing

## Overview
This repository contains the completed simulation models for the Monte Carlo Methods for Options Pricing project, sponsored by the Master of Science in Financial Engineering (MSFE) Director. The codebase utilizes Monte Carlo simulations to accurately price European, Asian, and American options under the Black-Scholes-Merton model. 

To optimize performance and heavily reduce computational time, the core numerical methods and algorithmic models have been implemented using C++ and Python.

## Implemented Models & Features

Our implementation covers all core requirements and optional advanced techniques:

### 1. European Vanilla Put Option
* **Standard Monte Carlo:** Prices the European put option using standard simulation.
* **Performance Tracking:** Outputs critical metrics including sample size, option price, estimated standard error, 95% confidence intervals, and total computational time in seconds.
* **Error Analysis:** Computes the absolute pricing error by benchmarking against the exact Black-Scholes analytical formula. 
* **Convergence Testing:** Investigates algorithm convergence as the sample size increases, achieving accuracy up to the cent.
* **Variance Reduction:** Implements the antithetic variate approach and compares its efficiency against the standard simulation.

### 2. Asian Call Option
* **Standard Monte Carlo:** Prices an Asian call based on the discretely monitored average asset price at maturity.
* **Performance Tracking:** Outputs the sample size, option price, estimated standard error, 95% confidence interval, and execution time.
* **Control Variates:** Uses a geometric Asian call option as a control variate, comparing the approach against standard methods to measure variance reduction effectiveness.
* **Moment Matching:** Implements the moment matching technique as an alternative pricing approach.

### 3. American Vanilla Put Option
* **Least-Squares Monte Carlo (LSM):** Prices American-style options (which allow for early exercise) using the Longstaff-Schwartz algorithm.
* **Parameter Sensitivity:** Evaluates how the approximate option price is impacted by varying the sample size ($N$), the number of time steps ($M$), and the number of basis regressors ($K$).
* **BBSR Benchmark:** Implements the Binomial Black-Scholes with Richardson Extrapolation (BBSR) method to generate a highly accurate benchmark price for validation.
* **Algorithmic Enhancements:** Incorporates findings from a broader literature review to implement promising enhancements to the standard Longstaff-Schwartz approach.

## Results & Documentation
The accompanying project report details the mathematical correctness of our results. Extensive tables and plots are provided to visualize pricing convergence, compare variance reduction techniques, and explain the specific efforts made to optimize computational execution times.
