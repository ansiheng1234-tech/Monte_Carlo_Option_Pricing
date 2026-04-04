import numpy as np

# S0 : initial stock price
# K  : strike price
# r  : risk-free rate (continuous compounding)
# q  : continuous dividend yield
# sigma : volatility
# T  : time to maturity
# ST : price at maturity
# n_steps : the number of the price generated within the total time
# dt = interval of time derived from T/n_steps

# Zij ​∼ N(0,1), i=1,2,…,N, j=1,2,...,n_steps
def get_Z(N, n_steps):
    Z = np.random.randn(N,n_steps)
    return Z

def get_dt(T, n_steps):
    dt = T/n_steps
    return dt

# log Sij =  (r - q - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z
# every step stock price log return under risk-neutral measure
def get_log_returns(r, q, sigma, dt, Z):
    log_return_matrix = (r - q - 0.5 * (sigma**2)) * dt + sigma * np.sqrt(dt) * Z
    return log_return_matrix

# turn the unit change of log return matrix into a price path
def get_S_matrix(S0, log_return_matrix):
    # add a column of zeros so that S0 also exists in here, otherwise the first column is already
    # doing geometric brownian motion, lefting out S0, but I'm curious if this is necessary
    N, n_steps = log_return_matrix.shape
    zero_column = np.zeros((N, 1))
    log_return_matrix = np.concatenate([zero_column, log_return_matrix], axis=1)
    S_matrix = np.cumsum(log_return_matrix, axis=1)
    S_matrix = S0*np.exp(S_matrix)
    return S_matrix

# add up each path and get the average price of the path
def get_average_ST(S_matrix):
    N, n_steps_plus1 = S_matrix.shape
    average_ST = (1/n_steps_plus1) * np.sum(S_matrix, axis=1)
    return average_ST 

# payoff, not discounted yet
def get_Asian_call_payoff(K, average_ST):
    Asian_call_payoff = np.maximum(average_ST - K, 0)
    return Asian_call_payoff 

def get_Asian_put_payoff(K, average_ST):
    Asian_put_payoff = np.maximum(K - average_ST, 0)
    return Asian_put_payoff 

# discount the payoff and get the expected value
def get_call_estimate(S0, K, r, q, sigma, T, N, n_steps):
    Z = get_Z(N, n_steps)
    dt = get_dt(T, n_steps)
    log_return_matrix = get_log_returns(r, q, sigma, dt, Z)
    S_matrix = get_S_matrix(S0, log_return_matrix)
    average_ST = get_average_ST(S_matrix)
    Asian_call_payoff = get_Asian_call_payoff(K, average_ST)
    discounted_payoff = np.exp(-r * T) * Asian_call_payoff
    estimate_price = (1/N) * np.sum(discounted_payoff, axis=0)
    return estimate_price

def get_put_estimate(S0, K, r, q, sigma, T, N, n_steps):
    Z = get_Z(N, n_steps)
    dt = get_dt(T, n_steps)
    log_return_matrix = get_log_returns(r, q, sigma, dt, Z)
    S_matrix = get_S_matrix(S0, log_return_matrix)
    average_ST = get_average_ST(S_matrix)
    Asian_put_payoff = get_Asian_put_payoff(K, average_ST)
    discounted_payoff = np.exp(-r * T) * Asian_put_payoff
    estimate_price = (1/N) * np.sum(discounted_payoff, axis=0)
    return estimate_price

# put together function only for convergence test use, return price and discounted payoff
def get_call_estimate_discounted_payoff(S0, K, r, q, sigma, T, N, n_steps):
    Z = get_Z(N, n_steps)
    dt = get_dt(T, n_steps)
    log_return_matrix = get_log_returns(r, q, sigma, dt, Z)
    S_matrix = get_S_matrix(S0, log_return_matrix)
    average_ST = get_average_ST(S_matrix)
    Asian_call_payoff = get_Asian_call_payoff(K, average_ST)
    discounted_payoff = np.exp(-r * T) * Asian_call_payoff
    estimate_price = (1/N) * np.sum(discounted_payoff, axis=0)
    return estimate_price, discounted_payoff