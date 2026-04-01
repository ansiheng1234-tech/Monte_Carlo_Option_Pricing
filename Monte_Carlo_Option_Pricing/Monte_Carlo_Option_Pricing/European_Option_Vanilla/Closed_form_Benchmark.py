import numpy as np 
from BSM_Option_Pricing import bsm_option_value, get_SE_CI_antithetic
import Monte_Carlo_Pricing

# S0 : initial stock price
# K  : strike price
# r  : risk-free rate (continuous compounding)
# q  : continuous dividend yield
# sigma : volatility
# T  : time to maturity
# ST : price at maturity

def get_SE_CI(S0, K, r, q, sigma, T, N):
    estimate_price, discounted_payoff = Monte_Carlo_Pricing.get_estimate_discounted_payoff(S0, K, r, q, sigma, T, N)
    SE = np.sqrt(1 / (N - 1) * np.sum((discounted_payoff - estimate_price)**2)) / np.sqrt(N)
    CI_low, CI_high = estimate_price - SE * 1.96, estimate_price + SE * 1.96
    return SE, CI_low, CI_high



S0 = 100
K = 100
T = 0.5
r = 0.04
q = 0.02
sigma = 0.2

N = 10000 # trials

results = get_SE_CI(S0, K, r, q, sigma, T, N)
print(f"Standard Error: {results[0]:.2f}")
print(f"95% Confidence Interval: [{results[1]:.2f}, {results[2]:.2f}]")

antithetic = get_SE_CI_antithetic(S0, K, r, q, sigma, T, N)
print(f"Standard Error, {antithetic[0]:.2f}")
print(f"95% Confidence Interval with antithetic approach: [{antithetic[1]:.2f}, {results[2]:.2f}]")
