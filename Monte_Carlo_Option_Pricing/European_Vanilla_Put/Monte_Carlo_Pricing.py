import numpy as np
from scipy.stats import norm 

# S0 : initial stock price
# K  : strike price
# r  : risk-free rate (continuous compounding)
# q  : continuous dividend yield
# sigma : volatility
# T  : time to maturity
# ST : price at maturity

def get_Z(N):
    Z = np.random.randn(N)
    return Z

def get_ST(S0, r, q, sigma, T, Z):
    ST = S0 * np.exp((r - q - 0.5 * (sigma**2)) * T + sigma * np.sqrt(T) * Z)
    return ST

def get_put_payoff(K, ST):
    payoff = np.maximum(K - ST, 0)
    return payoff

def get_estimate(S0, K, r, q, sigma, T, N):
    Z = get_Z(N)
    ST = get_ST(S0, r, q, sigma, T, Z)
    payoff = get_put_payoff(K, ST)
    discounted_payoff = np.exp(-r * T) * payoff
    estimate_price = (1 / N) * np.sum(discounted_payoff)
    return estimate_price

# a put together function only for closed form benchmark use
def get_estimate_discounted_payoff(S0, K, r, q, sigma, T, N):
    Z = get_Z(N)
    ST = get_ST(S0, r, q, sigma, T, Z)
    payoff = get_put_payoff(K, ST)
    discounted_payoff = np.exp(-r * T) * payoff
    estimate_price = (1 / N) * np.sum(discounted_payoff)
    return estimate_price, discounted_payoff




