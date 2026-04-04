from Monte_Carlo_Asian import get_call_estimate_discounted_payoff
from SE_CI import get_SE_CI
import matplotlib.pyplot as plt

# S0 : initial stock price
# K  : strike price
# r  : risk-free rate (continuous compounding)
# q  : continuous dividend yield
# sigma : volatility
# T  : time to maturity
# n_steps : the number of the price generated within the total time

S0 = 100
K = 100
r = 0.10
q = 0
sigma = 0.20
T = 1
n_steps = 50

Ns = [1000, 5000, 10000, 50000, 100000, 500000]

price_mc, dc_payoff = get_call_estimate_discounted_payoff(S0, K, r, q, sigma, T, Ns[0], n_steps)
SE, CI = get_SE_CI(price_mc, dc_payoff)

print(f"""
Sample Size: {Ns[0]}
Option Price: {price_mc:.2f}
Estimated Standard Error: {SE:.2f}
95% Confidence Interval: ({CI[0]:.2f}, {CI[1]:.2f})
""")

price_mc_list = []
SE_list = []
for N in Ns:
    price_mc, dc_payoff = get_call_estimate_discounted_payoff(S0, K, r, q, sigma, T, N, n_steps)
    SE, CI = get_SE_CI(price_mc, dc_payoff)
    price_mc_list.append(price_mc)
    SE_list.append(SE)


fig, axs = plt.subplots(1,2)
axs[0].plot(Ns, price_mc_list)
axs[0].set_xlabel("times of simulations")
axs[0].set_ylabel("estimated price by simulation")

axs[1].plot(Ns, SE_list)
axs[1].set_xlabel("times of simulations")
axs[1].set_ylabel("standard error")

plt.show()
