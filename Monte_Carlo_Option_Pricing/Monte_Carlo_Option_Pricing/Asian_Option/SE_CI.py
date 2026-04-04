import numpy as np 

# S0 : initial stock price
# K  : strike price
# r  : risk-free rate (continuous compounding)
# q  : continuous dividend yield
# sigma : volatility
# T  : time to maturity
# ST : price at maturity

# SE = std/sqrt(N)
# 95% CI = (price - SE * 1.96, price + SE * 1.96)
def get_SE_CI(estimate_price, discounted_payoff):
    N = len(discounted_payoff)
    SE = np.sqrt(1 / (N - 1) * np.sum((discounted_payoff - estimate_price)**2)) / np.sqrt(N)
    CI_low, CI_high = estimate_price - SE * 1.96, estimate_price + SE * 1.96
    return SE, (CI_low, CI_high)
