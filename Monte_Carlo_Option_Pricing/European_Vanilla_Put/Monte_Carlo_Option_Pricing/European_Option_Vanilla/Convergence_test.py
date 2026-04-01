from BSM_Option_Pricing import bsm_option_value
from European_Option_Vanilla.Monte_Carlo_European import get_estimate_discounted_payoff
from SE_CI import get_SE_CI
import matplotlib.pyplot as plt

# S0 : initial stock price
# K  : strike price
# r  : risk-free rate (continuous compounding)
# q  : continuous dividend yield
# sigma : volatility
# T  : time to maturity

#use sample  S0 = K = 100, T = 0.5, r = 0.04, q = 0.02, σ = 0.2
S0 = 100
K = 100
T = 0.5
r = 0.04
q = 0.02
sigma = 0.2

Ns = [1000, 5000, 10000, 50000, 100000, 500000]

price_bsm = bsm_option_value(S0, K, r, q, sigma, T, call = False)
price_mc, dc_payoff = get_estimate_discounted_payoff(S0, K, r, q, sigma, T, Ns[0])
SE, CI = get_SE_CI(price_mc, dc_payoff)
absolute_error = abs(price_mc - price_bsm)

print(f"""
Sample Size: {Ns[0]}
Option Price: {price_mc:.2f}
Estimated Standard Error: {SE:.2f}
95% Confidence Interval: ({CI[0]:.2f}, {CI[1]:.2f})
Absolute Pricing Error: {absolute_error:.2f}
""")
price_mc_list = []
absolute_error_list = []
SE_list = []
for N in Ns:
    price_mc, dc_payoff = get_estimate_discounted_payoff(S0, K, r, q, sigma, T, N)
    SE, CI = get_SE_CI(price_mc, dc_payoff)
    absolute_error = abs(price_mc - price_bsm)
    price_mc_list.append(price_mc)
    absolute_error_list.append(absolute_error)
    SE_list.append(SE)

fig, axs = plt.subplots(1,3)
axs[0].plot(Ns, price_mc_list)
axs[0].set_xlabel("times of simulations")
axs[0].set_ylabel("estimated price by simulation")

axs[1].plot(Ns, absolute_error_list)
axs[1].set_xlabel("times of simulations")
axs[1].set_ylabel("asolute pricing error")

axs[2].plot(Ns, SE_list)
axs[2].set_xlabel("times of simulations")
axs[2].set_ylabel("standard error")

plt.show()

    