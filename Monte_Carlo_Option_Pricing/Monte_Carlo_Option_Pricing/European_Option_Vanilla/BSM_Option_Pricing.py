import numpy as np 
from scipy.stats import norm

# S0 : initial stock price
# K  : strike price
# r  : risk-free rate (continuous compounding)
# q  : continuous dividend yield
# sigma : volatility
# T  : time to maturity

def bsm_option_value(S0, K, r, q, sigma, T, call = True):
    d1 = (np.log(S0/K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    C = S0 * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    P = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * np.exp(-q * T) * norm.cdf(-d1)

    if (call):
        return C
    return P

# We are using the variance reduction technique, antithetic approach,
# which generates N/2 samples z from N~(0,1) and generates N/2 samples -z. 
# We take the average of two paths to cancel out extreme outliers.

def get_SE_CI_antithetic(S0, K, r, q, sigma, T, N):

    z = np.random.standard_normal(N)

    drift = (r - q - 0.5 * sigma**2) * T
    vol_sqrt_t = sigma * np.sqrt(T)
    discount = np.exp(-r * T)
    
    ST1 = S0 * np.exp(drift + vol_sqrt_t * z)
    ST2 = S0 * np.exp(drift + vol_sqrt_t * (-z))

    payoff1 = np.maximum(K - ST1, 0) * discount
    payoff2 = np.maximum(K - ST2, 0) * discount

    antithetic_averages = (payoff1 + payoff2) / 2.0
    

    estimate_price = np.mean(antithetic_averages)
  
    SE = np.std(antithetic_averages, ddof=1) / np.sqrt(N)
    
    CI_low = estimate_price - 1.96 * SE
    CI_high = estimate_price + 1.96 * SE
    
    return SE, CI_low, CI_high

