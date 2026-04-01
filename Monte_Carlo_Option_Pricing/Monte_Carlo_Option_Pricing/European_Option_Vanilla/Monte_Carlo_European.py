import numpy as np

# S0 : initial stock price
# K  : strike price
# r  : risk-free rate (continuous compounding)
# q  : continuous dividend yield
# sigma : volatility
# T  : time to maturity
# ST : price at maturity

# Zi ​∼ N(0,1), i=1,2,…,N
# generate N random variable for simulation
def get_Z(N):
    Z = np.random.randn(N)
    return Z

# S_T = S_0 * exp((r - q - 0.5*sigma^2)*T + sigma*sqrt(T)*Z)
# terminal stock price under risk-neutral measure
def get_ST(S0, r, q, sigma, T, Z):
    ST = S0 * np.exp((r - q - 0.5 * (sigma**2)) * T + sigma * np.sqrt(T) * Z)
    return ST

# put payoff = max(K - S_T, 0), call is the opposite
def get_put_payoff(K, ST):
    payoff = np.maximum(K - ST, 0)
    return payoff

def get_call_payoff(K, ST):
    payoff = np.maximum(ST - K, 0)
    return payoff

# discount the payoff to current value, and then calculate the average of them
def get_put_estimate(S0, K, r, q, sigma, T, N):
    Z = get_Z(N)
    ST = get_ST(S0, r, q, sigma, T, Z)
    payoff = get_put_payoff(K, ST)
    discounted_payoff = np.exp(-r * T) * payoff
    estimate_price = (1 / N) * np.sum(discounted_payoff)
    return estimate_price

def get_call_estimate(S0, K, r, q, sigma, T, N):
    Z = get_Z(N)
    ST = get_ST(S0, r, q, sigma, T, Z)
    payoff = get_call_payoff(K, ST)
    discounted_payoff = np.exp(-r * T) * payoff
    estimate_price = (1 / N) * np.sum(discounted_payoff)
    return estimate_price

# a put together function only for closed form benchmark use, return price and discounted payoff
def get_put_estimate_discounted_payoff(S0, K, r, q, sigma, T, N):
    Z = get_Z(N)
    ST = get_ST(S0, r, q, sigma, T, Z)
    payoff = get_put_payoff(K, ST)
    discounted_payoff = np.exp(-r * T) * payoff
    estimate_price = (1 / N) * np.sum(discounted_payoff)
    return estimate_price, discounted_payoff




