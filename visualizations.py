import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__),
    "Monte_Carlo_Option_Pricing/Monte_Carlo_Option_Pricing"))

from European_Option_Vanilla.Monte_Carlo_European import (
    get_Z, get_ST, get_put_payoff, get_call_payoff,
    get_put_estimate_discounted_payoff,
)
from European_Option_Vanilla.SE_CI import get_SE_CI as european_se_ci
def bsm_option_value(S0, K, r, q, sigma, T, call=True):
    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    C = S0 * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    P = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * np.exp(-q * T) * norm.cdf(-d1)
    return C if call else P

def get_SE_CI_antithetic(S0, K, r, q, sigma, T, N):
    z = np.random.standard_normal(N)
    drift = (r - q - 0.5 * sigma**2) * T
    vol   = sigma * np.sqrt(T)
    disc  = np.exp(-r * T)
    p1 = np.maximum(K - S0 * np.exp(drift + vol *  z), 0) * disc
    p2 = np.maximum(K - S0 * np.exp(drift + vol * -z), 0) * disc
    av = (p1 + p2) / 2.0
    price = np.mean(av)
    SE    = np.std(av, ddof=1) / np.sqrt(N)
    return SE, price - 1.96 * SE, price + 1.96 * SE
from Asian_Option.Monte_Carlo_Asian import (
    get_Z as asian_get_Z, get_dt, get_log_returns, get_S_matrix,
    get_average_ST, get_Asian_call_payoff, get_call_estimate_discounted_payoff,
)
from Asian_Option.SE_CI import get_SE_CI as asian_se_ci
from American_Option.Monte_Carlo_American import (
    get_Z as american_get_Z, get_S_matrix as american_get_S_matrix,
    get_log_returns as american_get_log_returns, get_dt as american_get_dt,
    get_regression_matrix, get_American_put_estimate,
)

# ── Shared parameters ──────────────────────────────────────────────────────────
S0    = 100
K     = 100
r     = 0.04
q     = 0.02
sigma = 0.2
T     = 0.5

ASIAN_r     = 0.10
ASIAN_q     = 0.0
ASIAN_sigma = 0.20
ASIAN_T     = 1.0
ASIAN_n_steps = 50

N_PATHS   = 50      # paths shown in spaghetti plots
N_SIM     = 50_000  # default simulation count
Ns        = [1_000, 5_000, 10_000, 50_000, 100_000, 500_000]

# ── Plot 1: GBM Simulated Price Paths (European / American) ───────────────────
n_steps_plot = 100
dt_plot = T / n_steps_plot
t_grid  = np.linspace(0, T, n_steps_plot + 1)

Z_paths = np.random.randn(N_PATHS, n_steps_plot)
log_ret = (r - q - 0.5 * sigma**2) * dt_plot + sigma * np.sqrt(dt_plot) * Z_paths
zero_col = np.zeros((N_PATHS, 1))
log_ret  = np.concatenate([zero_col, log_ret], axis=1)
S_paths  = S0 * np.exp(np.cumsum(log_ret, axis=1))   # shape (N_PATHS, n_steps+1)

itm_put  = S_paths[:, -1] < K   # paths that end in-the-money for a put

fig1, ax1 = plt.subplots(figsize=(10, 5))
for i in range(N_PATHS):
    color = "#d62728" if itm_put[i] else "#aec7e8"
    ax1.plot(t_grid, S_paths[i], color=color, linewidth=0.8, alpha=0.7)

ax1.axhline(K, color="black", linewidth=1.5, linestyle="--", label=f"Strike K = {K}")
ax1.set_xlabel("Time (years)")
ax1.set_ylabel("Stock Price S(t)")
ax1.set_title("European Option: Simulated GBM Price Paths\n"
              "(red = ITM at maturity for put, blue = OTM)")
ax1.legend()
fig1.tight_layout()

# ── Plot 2: Asian Option Paths with Running Average ───────────────────────────
n_steps_asian_plot = ASIAN_n_steps
dt_asian = ASIAN_T / n_steps_asian_plot
t_grid_asian = np.linspace(0, ASIAN_T, n_steps_asian_plot + 1)

Z_asian = np.random.randn(N_PATHS, n_steps_asian_plot)
log_ret_asian = (ASIAN_r - ASIAN_q - 0.5 * ASIAN_sigma**2) * dt_asian + ASIAN_sigma * np.sqrt(dt_asian) * Z_asian
zero_col_asian = np.zeros((N_PATHS, 1))
log_ret_asian  = np.concatenate([zero_col_asian, log_ret_asian], axis=1)
S_asian = S0 * np.exp(np.cumsum(log_ret_asian, axis=1))   # shape (N_PATHS, n_steps+1)

running_avg = np.cumsum(S_asian, axis=1) / np.arange(1, n_steps_asian_plot + 2)
itm_asian   = running_avg[:, -1] > K   # paths ITM for Asian call

fig2, ax2 = plt.subplots(figsize=(10, 5))
for i in range(N_PATHS):
    color = "#d62728" if itm_asian[i] else "#aec7e8"
    ax2.plot(t_grid_asian, S_asian[i],      color=color,    linewidth=0.7, alpha=0.5)
    ax2.plot(t_grid_asian, running_avg[i],  color="#2ca02c", linewidth=1.0, alpha=0.6)

ax2.axhline(K, color="black", linewidth=1.5, linestyle="--", label=f"Strike K = {K}")
ax2.plot([], [], color="#aec7e8",  linewidth=1.5, label="Stock path (OTM call)")
ax2.plot([], [], color="#d62728",  linewidth=1.5, label="Stock path (ITM call)")
ax2.plot([], [], color="#2ca02c",  linewidth=1.5, label="Running arithmetic average")
ax2.set_xlabel("Time (years)")
ax2.set_ylabel("Price")
ax2.set_title("Asian Option: Simulated Paths with Running Arithmetic Average\n"
              "(green = running average, red = ITM at maturity for call)")
ax2.legend()
fig2.tight_layout()

# ── Plot 3: Terminal Price Distribution vs Lognormal ─────────────────────────
Z_term = np.random.randn(N_SIM)
ST_sim = S0 * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z_term)

mu_ln  = np.log(S0) + (r - q - 0.5 * sigma**2) * T
s_ln   = sigma * np.sqrt(T)
x_range = np.linspace(ST_sim.min(), ST_sim.max(), 500)
pdf_ln  = (1 / (x_range * s_ln * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((np.log(x_range) - mu_ln) / s_ln)**2)

fig3, ax3 = plt.subplots(figsize=(9, 5))
ax3.hist(ST_sim, bins=100, density=True, color="#aec7e8", edgecolor="white",
         linewidth=0.3, label="Simulated $S_T$")
ax3.plot(x_range, pdf_ln, color="#d62728", linewidth=2, label="Theoretical lognormal PDF")
ax3.axvline(K, color="black", linewidth=1.5, linestyle="--", label=f"Strike K = {K}")
ax3.set_xlabel("Terminal Stock Price $S_T$")
ax3.set_ylabel("Density")
ax3.set_title(f"Terminal Price Distribution vs Theoretical Lognormal\n"
              f"(N = {N_SIM:,}, S0={S0}, σ={sigma}, T={T})")
ax3.legend()
fig3.tight_layout()

# ── Plot 4: Discounted Payoff Distributions ───────────────────────────────────
Z4 = np.random.randn(N_SIM)
ST4 = S0 * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z4)
disc = np.exp(-r * T)

eu_put_payoffs  = disc * np.maximum(K - ST4, 0)
eu_call_payoffs = disc * np.maximum(ST4 - K, 0)

Z4_asian = np.random.randn(N_SIM, ASIAN_n_steps)
lr4 = (ASIAN_r - ASIAN_q - 0.5*ASIAN_sigma**2)*get_dt(ASIAN_T, ASIAN_n_steps) + ASIAN_sigma*np.sqrt(get_dt(ASIAN_T, ASIAN_n_steps))*Z4_asian
S4_asian = S0 * np.exp(np.cumsum(np.concatenate([np.zeros((N_SIM,1)), lr4], axis=1), axis=1))
avg4 = np.mean(S4_asian, axis=1)
asian_call_payoffs = np.exp(-ASIAN_r * ASIAN_T) * np.maximum(avg4 - K, 0)

fig4, axes4 = plt.subplots(1, 3, figsize=(15, 5))

for ax, payoffs, label, color in zip(
    axes4,
    [eu_put_payoffs, eu_call_payoffs, asian_call_payoffs],
    ["European Put", "European Call", "Asian Call"],
    ["#aec7e8", "#ffbb78", "#98df8a"],
):
    price = np.mean(payoffs)
    ax.hist(payoffs, bins=80, density=True, color=color, edgecolor="white", linewidth=0.3)
    ax.axvline(price, color="#d62728", linewidth=2, linestyle="--",
               label=f"MC Price = {price:.3f}")
    ax.set_xlabel("Discounted Payoff")
    ax.set_ylabel("Density")
    ax.set_title(f"{label} Payoff Distribution")
    ax.legend()

fig4.suptitle(f"Discounted Payoff Distributions  (N = {N_SIM:,})", fontsize=13)
fig4.tight_layout()

# ── Plot 5: European Convergence (price, error, SE) ──────────────────────────
bsm_put = bsm_option_value(S0, K, r, q, sigma, T, call=False)

prices5, errors5, ses5, ci_lo5, ci_hi5 = [], [], [], [], []
for N in Ns:
    price, dpayoff = get_put_estimate_discounted_payoff(S0, K, r, q, sigma, T, N)
    se, ci = european_se_ci(price, dpayoff)
    prices5.append(price)
    errors5.append(abs(price - bsm_put))
    ses5.append(se)
    ci_lo5.append(ci[0])
    ci_hi5.append(ci[1])

ref_se = ses5[0] * np.sqrt(Ns[0]) / np.sqrt(np.array(Ns))

fig5, axes5 = plt.subplots(1, 3, figsize=(15, 5))

# price convergence with CI band
axes5[0].fill_between(Ns, ci_lo5, ci_hi5, alpha=0.25, color="#aec7e8", label="95% CI")
axes5[0].plot(Ns, prices5, color="#1f77b4", linewidth=2, marker="o", markersize=4, label="MC Price")
axes5[0].axhline(bsm_put, color="#d62728", linewidth=1.5, linestyle="--", label=f"BSM = {bsm_put:.4f}")
axes5[0].set_xscale("log")
axes5[0].set_xlabel("Number of Simulations (log scale)")
axes5[0].set_ylabel("Put Price")
axes5[0].set_title("MC Price Convergence to BSM")
axes5[0].legend()

# absolute error
axes5[1].plot(Ns, errors5, color="#ff7f0e", linewidth=2, marker="o", markersize=4, label="|MC - BSM|")
axes5[1].set_xscale("log")
axes5[1].set_yscale("log")
axes5[1].set_xlabel("Number of Simulations (log scale)")
axes5[1].set_ylabel("Absolute Error (log scale)")
axes5[1].set_title("Absolute Pricing Error vs N")
axes5[1].legend()

# SE decay with 1/sqrt(N) reference
axes5[2].plot(Ns, ses5, color="#2ca02c", linewidth=2, marker="o", markersize=4, label="Standard Error")
axes5[2].plot(Ns, ref_se, color="black", linewidth=1.5, linestyle="--", label="1/√N reference")
axes5[2].set_xscale("log")
axes5[2].set_yscale("log")
axes5[2].set_xlabel("Number of Simulations (log scale)")
axes5[2].set_ylabel("Standard Error (log scale)")
axes5[2].set_title("SE Decay vs N")
axes5[2].legend()

fig5.suptitle("European Put — Monte Carlo Convergence Analysis", fontsize=13)
fig5.tight_layout()

# ── Plot 6: Asian Convergence (price, SE) ────────────────────────────────────
prices6, ses6, ci_lo6, ci_hi6 = [], [], [], []
for N in Ns:
    price, dpayoff = get_call_estimate_discounted_payoff(S0, K, ASIAN_r, ASIAN_q, ASIAN_sigma, ASIAN_T, N, ASIAN_n_steps)
    se, ci = asian_se_ci(price, dpayoff)
    prices6.append(price)
    ses6.append(se)
    ci_lo6.append(ci[0])
    ci_hi6.append(ci[1])

ref_se6 = ses6[0] * np.sqrt(Ns[0]) / np.sqrt(np.array(Ns))

fig6, axes6 = plt.subplots(1, 2, figsize=(12, 5))

axes6[0].fill_between(Ns, ci_lo6, ci_hi6, alpha=0.25, color="#98df8a", label="95% CI")
axes6[0].plot(Ns, prices6, color="#2ca02c", linewidth=2, marker="o", markersize=4, label="MC Price")
axes6[0].set_xscale("log")
axes6[0].set_xlabel("Number of Simulations (log scale)")
axes6[0].set_ylabel("Call Price")
axes6[0].set_title("Asian Call — MC Price Convergence\n(no closed-form benchmark)")
axes6[0].legend()

axes6[1].plot(Ns, ses6, color="#2ca02c", linewidth=2, marker="o", markersize=4, label="Standard Error")
axes6[1].plot(Ns, ref_se6, color="black", linewidth=1.5, linestyle="--", label="1/√N reference")
axes6[1].set_xscale("log")
axes6[1].set_yscale("log")
axes6[1].set_xlabel("Number of Simulations (log scale)")
axes6[1].set_ylabel("Standard Error (log scale)")
axes6[1].set_title("Asian Call — SE Decay vs N")
axes6[1].legend()

fig6.suptitle("Asian Call — Monte Carlo Convergence Analysis", fontsize=13)
fig6.tight_layout()

# ── Plot 7: American Convergence (price) ─────────────────────────────────────
# LSM is expensive — use a smaller N set and fewer steps
Ns_american  = [1_000, 2_000, 5_000, 10_000, 25_000]
AM_n_steps   = 50
AM_n_reg     = 3
AM_r, AM_q   = 0.04, 0.02

prices7 = []
for N in Ns_american:
    p = get_American_put_estimate(S0, K, AM_r, AM_q, sigma, T, N, AM_n_steps, AM_n_reg)
    prices7.append(p)

eu_put_bsm = bsm_option_value(S0, K, AM_r, AM_q, sigma, T, call=False)

fig7, ax7 = plt.subplots(figsize=(9, 5))
ax7.plot(Ns_american, prices7, color="#9467bd", linewidth=2, marker="o",
         markersize=5, label="American Put (LSM)")
ax7.axhline(eu_put_bsm, color="#d62728", linewidth=1.5, linestyle="--",
            label=f"European Put BSM = {eu_put_bsm:.4f}")
ax7.fill_between(Ns_american,
                 [p - 0.02 for p in prices7],
                 [p + 0.02 for p in prices7],
                 alpha=0.15, color="#9467bd", label="±0.02 band")
ax7.set_xscale("log")
ax7.set_xlabel("Number of Simulations (log scale)")
ax7.set_ylabel("Put Price")
ax7.set_title("American Put (LSM) — Price Convergence\n"
              "Early exercise premium = American − European")
ax7.legend()
fig7.tight_layout()

# ── Plot 8: Antithetic Variance Reduction ────────────────────────────────────
ses8_std, ses8_anti = [], []
for N in Ns:
    # standard MC SE (reuse european_se_ci)
    price_s, dp_s = get_put_estimate_discounted_payoff(S0, K, r, q, sigma, T, N)
    se_s, _ = european_se_ci(price_s, dp_s)
    ses8_std.append(se_s)

    # antithetic SE
    se_a, _, _ = get_SE_CI_antithetic(S0, K, r, q, sigma, T, N)
    ses8_anti.append(se_a)

vr_ratio = [s / a for s, a in zip(ses8_std, ses8_anti)]   # > 1 means antithetic is better
ref_se8  = ses8_std[0] * np.sqrt(Ns[0]) / np.sqrt(np.array(Ns))

fig8, axes8 = plt.subplots(1, 2, figsize=(13, 5))

axes8[0].plot(Ns, ses8_std,  color="#1f77b4", linewidth=2, marker="o", markersize=4, label="Standard MC")
axes8[0].plot(Ns, ses8_anti, color="#d62728", linewidth=2, marker="o", markersize=4, label="Antithetic MC")
axes8[0].plot(Ns, ref_se8,   color="black",   linewidth=1.5, linestyle="--", label="1/√N reference")
axes8[0].set_xscale("log")
axes8[0].set_yscale("log")
axes8[0].set_xlabel("Number of Simulations (log scale)")
axes8[0].set_ylabel("Standard Error (log scale)")
axes8[0].set_title("Standard vs Antithetic SE")
axes8[0].legend()

axes8[1].plot(Ns, vr_ratio, color="#2ca02c", linewidth=2, marker="o", markersize=5)
axes8[1].axhline(1.0, color="black", linewidth=1, linestyle="--", label="No improvement (ratio = 1)")
axes8[1].set_xscale("log")
axes8[1].set_xlabel("Number of Simulations (log scale)")
axes8[1].set_ylabel("SE ratio (Standard / Antithetic)")
axes8[1].set_title("Variance Reduction Factor\n(higher = antithetic is better)")
axes8[1].legend()

fig8.suptitle("Antithetic Variates — Variance Reduction for European Put", fontsize=13)
fig8.tight_layout()

# ── Plot 9: Parameter Sensitivity ────────────────────────────────────────────
N_SENS   = 20_000
N_SENS_AM = 8_000

# 9a: price vs volatility
sigmas9 = np.linspace(0.05, 0.60, 20)
mc_put_sig, bsm_put_sig = [], []
mc_call_sig, bsm_call_sig = [], []
for sig in sigmas9:
    p, dp = get_put_estimate_discounted_payoff(S0, K, r, q, sig, T, N_SENS)
    mc_put_sig.append(p)
    bsm_put_sig.append(bsm_option_value(S0, K, r, q, sig, T, call=False))
    Z9 = np.random.randn(N_SENS)
    ST9 = S0 * np.exp((r - q - 0.5*sig**2)*T + sig*np.sqrt(T)*Z9)
    mc_call_sig.append(np.exp(-r*T) * np.mean(np.maximum(ST9 - K, 0)))
    bsm_call_sig.append(bsm_option_value(S0, K, r, q, sig, T, call=True))

# 9b: price vs spot (moneyness)
S0s9 = np.linspace(70, 130, 20)
mc_put_s0, bsm_put_s0 = [], []
am_put_s0, eu_put_s0_bsm = [], []
for s in S0s9:
    p, dp = get_put_estimate_discounted_payoff(s, K, r, q, sigma, T, N_SENS)
    mc_put_s0.append(p)
    bsm_put_s0.append(bsm_option_value(s, K, r, q, sigma, T, call=False))
    am_put_s0.append(get_American_put_estimate(s, K, AM_r, AM_q, sigma, T, N_SENS_AM, AM_n_steps, AM_n_reg))
    eu_put_s0_bsm.append(bsm_option_value(s, K, AM_r, AM_q, sigma, T, call=False))

fig9, axes9 = plt.subplots(1, 2, figsize=(14, 5))

# volatility sensitivity
axes9[0].plot(sigmas9, mc_put_sig,  color="#1f77b4", linewidth=2, marker="o", markersize=3, label="MC Put")
axes9[0].plot(sigmas9, bsm_put_sig, color="#1f77b4", linewidth=1.5, linestyle="--", label="BSM Put")
axes9[0].plot(sigmas9, mc_call_sig,  color="#ff7f0e", linewidth=2, marker="o", markersize=3, label="MC Call")
axes9[0].plot(sigmas9, bsm_call_sig, color="#ff7f0e", linewidth=1.5, linestyle="--", label="BSM Call")
axes9[0].set_xlabel("Volatility σ")
axes9[0].set_ylabel("Option Price")
axes9[0].set_title("European Put & Call Price vs Volatility\n(solid = MC, dashed = BSM)")
axes9[0].legend()

# moneyness: American vs European
premium = [a - e for a, e in zip(am_put_s0, eu_put_s0_bsm)]
axes9[1].plot(S0s9, am_put_s0,    color="#9467bd", linewidth=2, label="American Put (LSM)")
axes9[1].plot(S0s9, eu_put_s0_bsm, color="#d62728", linewidth=2, linestyle="--", label="European Put (BSM)")
axes9[1].fill_between(S0s9, eu_put_s0_bsm, am_put_s0, alpha=0.2, color="#9467bd", label="Early exercise premium")
axes9[1].axvline(K, color="black", linewidth=1, linestyle=":", label=f"ATM (S0=K={K})")
axes9[1].set_xlabel("Initial Stock Price S0")
axes9[1].set_ylabel("Put Price")
axes9[1].set_title("American vs European Put — Moneyness\n(shaded = early exercise premium)")
axes9[1].legend()

fig9.suptitle("Parameter Sensitivity", fontsize=13)
fig9.tight_layout()

# ── Plot 10: American Early Exercise Boundary ────────────────────────────────
def get_exercise_boundary(S0, K, r, q, sigma, T, N, n_steps, n_regressors):
    dt = T / n_steps
    Z = np.random.randn(N, n_steps)
    log_ret = (r - q - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z
    S_mat = S0 * np.exp(np.cumsum(np.concatenate([np.zeros((N,1)), log_ret], axis=1), axis=1))

    cashflow = np.maximum(K - S_mat[:, -1], 0)
    boundary = np.full(n_steps + 1, np.nan)
    boundary[-1] = K

    for t in range(n_steps - 1, 0, -1):
        disc_cf = np.exp(-r * dt) * cashflow
        ex_val  = np.maximum(K - S_mat[:, t], 0)
        itm     = np.where(ex_val > 0)[0]

        if len(itm) < n_regressors:
            cashflow = disc_cf
            continue

        x = S_mat[itm, t]
        y = disc_cf[itm]
        X = np.column_stack([x**k for k in range(n_regressors)])
        beta = np.linalg.lstsq(X, y, rcond=None)[0]

        # find critical S* where exercise value = continuation value
        s_grid  = np.linspace(max(x.min(), K * 0.5), K, 300)
        X_grid  = np.column_stack([s_grid**k for k in range(n_regressors)])
        diff    = (K - s_grid) - (X_grid @ beta)
        itm_opt = np.where(diff > 0)[0]
        if len(itm_opt) > 0:
            boundary[t] = s_grid[itm_opt[-1]]

        cont = X @ beta
        new_cf = disc_cf.copy()
        new_cf[itm[ex_val[itm] > cont]] = ex_val[itm[ex_val[itm] > cont]]
        cashflow = new_cf

    return boundary

t_grid10 = np.linspace(0, T, AM_n_steps + 1)
boundary10 = get_exercise_boundary(S0, K, AM_r, AM_q, sigma, T, 60_000, AM_n_steps, AM_n_reg)

fig10, ax10 = plt.subplots(figsize=(9, 5))
ax10.plot(t_grid10, boundary10, color="#9467bd", linewidth=2, label="Early exercise boundary S*(t)")
ax10.axhline(K, color="black", linewidth=1.5, linestyle="--", label=f"Strike K = {K}")
ax10.fill_between(t_grid10, boundary10, K, where=(~np.isnan(boundary10)).tolist(),
                  alpha=0.15, color="#9467bd", label="Exercise region (S < S*)")
ax10.set_xlabel("Time (years)")
ax10.set_ylabel("Stock Price")
ax10.set_title("American Put — Early Exercise Boundary\n"
               "Exercise immediately when S(t) falls below S*(t)")
ax10.legend()
fig10.tight_layout()

# ── Plot 11: LSM Regression Fit at One Time Step ─────────────────────────────
TARGET_T = AM_n_steps // 2   # capture regression at the midpoint time step

N11   = 30_000
dt11  = T / AM_n_steps
Z11   = np.random.randn(N11, AM_n_steps)
log11 = (AM_r - AM_q - 0.5*sigma**2)*dt11 + sigma*np.sqrt(dt11)*Z11
S11   = S0 * np.exp(np.cumsum(np.concatenate([np.zeros((N11,1)), log11], axis=1), axis=1))

cashflow11 = np.maximum(K - S11[:, -1], 0)
captured   = {}   # will hold data at TARGET_T

for t in range(AM_n_steps - 1, 0, -1):
    disc_cf = np.exp(-AM_r * dt11) * cashflow11
    ex_val  = np.maximum(K - S11[:, t], 0)
    itm     = np.where(ex_val > 0)[0]

    if len(itm) < AM_n_reg:
        cashflow11 = disc_cf
        continue

    x   = S11[itm, t]
    y   = disc_cf[itm]
    X11 = np.column_stack([x**k for k in range(AM_n_reg)])
    beta11 = np.linalg.lstsq(X11, y, rcond=None)[0]
    cont11 = X11 @ beta11

    if t == TARGET_T:
        captured = {"x": x, "y": y, "beta": beta11, "t": t * dt11}

    new_cf = disc_cf.copy()
    new_cf[itm[ex_val[itm] > cont11]] = ex_val[itm[ex_val[itm] > cont11]]
    cashflow11 = new_cf

# plot the captured regression
x_cap, y_cap, beta_cap = captured["x"], captured["y"], captured["beta"]
s_fit   = np.linspace(x_cap.min(), K, 300)
X_fit   = np.column_stack([s_fit**k for k in range(AM_n_reg)])
cont_fit = X_fit @ beta_cap
ex_fit   = K - s_fit

# find S* (crossover)
diff11   = ex_fit - cont_fit
cross    = np.where(diff11 > 0)[0]
s_star   = s_fit[cross[-1]] if len(cross) > 0 else None

fig11, ax11 = plt.subplots(figsize=(9, 5))
ax11.scatter(x_cap, y_cap, color="steelblue", alpha=0.15, s=6, label="ITM paths: (S_t, discounted future CF)")
ax11.plot(s_fit, cont_fit, color="#d62728", linewidth=2.5, label="Continuation value (fitted polynomial)")
ax11.plot(s_fit, ex_fit,   color="#2ca02c", linewidth=2.5, linestyle="--", label="Immediate exercise value (K − S)")
if s_star is not None:
    ax11.axvline(s_star, color="black", linewidth=1.5, linestyle=":",
                 label=f"S* = {s_star:.1f}  (exercise boundary)")
ax11.set_xlabel("Stock Price S(t)")
ax11.set_ylabel("Value")
ax11.set_title(f"LSM Regression at t = {captured.get('t', 0):.2f} yrs  (midpoint of [{0}, {T}])\n"
               "Left of S*: exercise now   |   Right of S*: hold")
ax11.legend()
ax11.set_xlim(x_cap.min(), K + 2)
ax11.set_ylim(-1, np.percentile(y_cap, 98) * 1.3)
fig11.tight_layout()

plt.show()
