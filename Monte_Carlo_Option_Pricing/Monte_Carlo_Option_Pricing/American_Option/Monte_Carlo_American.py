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
    Z = np.random.randn(N, n_steps)
    return Z

def get_dt(T, n_steps):
    dt = T / n_steps
    return dt

# log Sij =  (r - q - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z
# every step stock price log return under risk-neutral measure
def get_log_returns(r, q, sigma, dt, Z):
    log_return_matrix = (r - q - 0.5 * (sigma**2)) * dt + sigma * np.sqrt(dt) * Z
    return log_return_matrix

# turn the unit change of log return matrix into a price path
def get_S_matrix(S0, log_return_matrix):
    # add a column of zeros so that S0 also exists in here
    N, n_steps = log_return_matrix.shape
    zero_column = np.zeros((N, 1))
    log_return_matrix = np.concatenate([zero_column, log_return_matrix], axis=1)
    S_matrix = np.cumsum(log_return_matrix, axis=1)
    S_matrix = S0 * np.exp(S_matrix)
    return S_matrix

# former part same as Asian, copied directly
# create [1, x, x^2, ..., x^(n_regressors-1)]
def get_regression_matrix(x, n_regressors):
    X = np.column_stack([x**k for k in range(n_regressors)])
    return X

# estimate American put price
def get_American_put_estimate(S0, K, r, q, sigma, T, N, n_steps, n_regressors):
    # Step 1: simulate stock price paths
    Z = get_Z(N, n_steps)
    dt = get_dt(T, n_steps)
    log_return_matrix = get_log_returns(r, q, sigma, dt, Z)
    S_matrix = get_S_matrix(S0, log_return_matrix)
    # payoff at maturity
    # each path gets payoff at the last column
    cashflow = np.maximum(K - S_matrix[:, -1], 0)
    # move backward in time, t = n_steps-1 down to 1
    for t in range(n_steps - 1, 0, -1):
        discounted_cashflow = np.exp(-r * dt) * cashflow
        exercise_value = np.maximum(K - S_matrix[:, t], 0)
        # find in-the-money paths 
        itm_indices = np.where(exercise_value > 0)[0]
        # if no ITM paths, just keep discounting
        if len(itm_indices) == 0:
            cashflow = discounted_cashflow
            continue
        # x = current prices of ITM paths
        x = S_matrix[itm_indices, t]
        # y = future payoff discounted to time t
        y = discounted_cashflow[itm_indices]

        X = get_regression_matrix(x, n_regressors)
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        continuation_value = X @ beta

        exercise_now = exercise_value[itm_indices] > continuation_value
        new_cashflow = discounted_cashflow.copy()
        exercise_now_indices = itm_indices[exercise_now]
        new_cashflow[exercise_now_indices] = exercise_value[exercise_now_indices]
        cashflow = new_cashflow

    # discount back to time 0
    estimate_price = np.exp(-r * dt) * np.mean(cashflow)

    return estimate_price