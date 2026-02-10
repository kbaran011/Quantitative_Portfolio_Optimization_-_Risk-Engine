import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf

tickers = ['VOO', 'JPM', 'AMZN', 'META', 'TSLA']

start_date = '2020-01-01'
end_date = '2025-01-01'

prices = yf.download(tickers, start=start_date, end=end_date)['Close']
print(prices.head())

prices.plot(title='Asset Prices Over Time')
plt.show()

# 1. Calculate Returns
returns = prices.pct_change()

# 2. REPLACE INFINITY
# pct_change() can create 'inf' if a price is 0. dropna() ignores 'inf'.
returns.replace([np.inf, -np.inf], np.nan, inplace=True)

# 3. DROP MISSING VALUES
returns.dropna(inplace=True)

# 4. FORCE TYPE CONVERSION
# Ensure data is strictly float64 (avoids object/string errors)
returns = returns.astype(np.float64)

print('\nFirst 5 rows of daily returns:\n', returns.head())

mean_daily_returns = returns.mean()
print('\nMean daily returns:\n', mean_daily_returns.head())
print(mean_daily_returns)

cov_matrix = returns.cov()
print('\nCovariance matrix:\n', cov_matrix)

#Not every day is a trading day, so we need to annualize the returns and covariance matrix
trading_days = 252
annualized_returns = mean_daily_returns * trading_days
annualized_cov_matrix = cov_matrix * trading_days

def portfolio_performance(weights, mean_daily_returns, cov_matrix, trading_days=252):
    """
    Returns annualized (expected_return, volatility) for a portfolio.
    - mean_daily_returns: expected daily returns for each asset
    - cov_matrix: daily covariance matrix of returns
    """
    # annualized expected return
    port_return = np.sum(mean_daily_returns * weights) * trading_days

    # annualized volatility
    port_var = weights.T @ cov_matrix @ weights
    port_vol = np.sqrt(port_var) * np.sqrt(trading_days)

    return port_return, port_vol

 # -------------------------------------------
# Step 3: Monte Carlo simulation (Efficient Frontier)
# -------------------------------------------
num_portfolios = 10000
risk_free_rate = 0.00  # set to 0 for now; we can adjust later

results = np.zeros((3, num_portfolios))              # rows: return, vol, sharpe
weights_record = np.zeros((len(tickers), num_portfolios))  # store weights for each portfolio

for i in range(num_portfolios):
    w = np.random.random(len(tickers))
    w /= np.sum(w)  # weights sum to 1

    weights_record[:, i] = w

    port_return, port_vol = portfolio_performance(
        w, mean_daily_returns, cov_matrix, trading_days=trading_days
    )

    sharpe = (port_return - risk_free_rate) / port_vol if port_vol != 0 else np.nan

    results[0, i] = port_return
    results[1, i] = port_vol
    results[2, i] = sharpe

results_df = pd.DataFrame(
    results.T,
    columns=["Return", "Volatility", "Sharpe"]
)

print("\nSimulation results (first 5 rows):")
print(results_df.head())

# Plot efficient frontier
plt.figure(figsize=(10, 6))
plt.scatter(results_df["Volatility"], results_df["Return"], alpha=0.3)
plt.xlabel("Annualized Volatility")
plt.ylabel("Annualized Expected Return")
plt.title("Efficient Frontier (Monte Carlo)")
plt.show()

# -------------------------------------------
# Step 4: Identify best portfolios
# -------------------------------------------
max_sharpe_idx = results_df["Sharpe"].idxmax()
min_var_idx = results_df["Volatility"].idxmin()

max_sharpe_weights = weights_record[:, max_sharpe_idx]
min_var_weights = weights_record[:, min_var_idx]

print("\n=== Max Sharpe Portfolio ===")
print(f"Return:     {results_df.loc[max_sharpe_idx, 'Return']:.4f}")
print(f"Volatility: {results_df.loc[max_sharpe_idx, 'Volatility']:.4f}")
print(f"Sharpe:     {results_df.loc[max_sharpe_idx, 'Sharpe']:.4f}")
print(pd.Series(max_sharpe_weights, index=tickers).sort_values(ascending=False))

print("\n=== Min Variance Portfolio ===")
print(f"Return:     {results_df.loc[min_var_idx, 'Return']:.4f}")
print(f"Volatility: {results_df.loc[min_var_idx, 'Volatility']:.4f}")
print(f"Sharpe:     {results_df.loc[min_var_idx, 'Sharpe']:.4f}")
print(pd.Series(min_var_weights, index=tickers).sort_values(ascending=False))

# Plot with highlighted portfolios
plt.figure(figsize=(10, 6))
plt.scatter(results_df["Volatility"], results_df["Return"], alpha=0.25)

plt.scatter(
    results_df.loc[max_sharpe_idx, "Volatility"],
    results_df.loc[max_sharpe_idx, "Return"],
    marker="*",
    s=250,
    label="Max Sharpe"
)

plt.scatter(
    results_df.loc[min_var_idx, "Volatility"],
    results_df.loc[min_var_idx, "Return"],
    marker="o",
    s=120,
    label="Min Variance"
)

plt.xlabel("Annualized Volatility")
plt.ylabel("Annualized Expected Return")
plt.title("Efficient Frontier with Optimal Portfolios")
plt.legend()
plt.show()

# -------------------------------------------
# Step 5: Monte Carlo VaR / CVaR (1-day)
# -------------------------------------------
np.random.seed(42)

n_simulations = 20000
alpha_95 = 0.05  # 95% VaR => worst 5% tail
alpha_99 = 0.01  # 99% VaR => worst 1% tail

# Simulate 1-day asset returns (multivariate normal baseline)
# (Make mean/cov explicitly float64, symmetric, and numerically stable)
mu = np.asarray(mean_daily_returns.values, dtype=np.float64)

Sigma = np.asarray(cov_matrix.values, dtype=np.float64)
Sigma = 0.5 * (Sigma + Sigma.T)  # force symmetry
Sigma = Sigma + np.eye(Sigma.shape[0], dtype=np.float64) * 1e-12  # tiny jitter for stability

sim_asset_returns = np.random.multivariate_normal(
    mean=mu,
    cov=Sigma,
    size=n_simulations
).astype(np.float64, copy=False)

# Convert to 1-day portfolio returns using Max Sharpe weights
w_ms = np.asarray(max_sharpe_weights, dtype=np.float64).reshape(-1)
sim_port_returns = sim_asset_returns.dot(w_ms)

# Safety: drop any rare non-finite simulations (should be ~0 rows)
mask_finite = np.isfinite(sim_port_returns)
if not mask_finite.all():
    sim_port_returns = sim_port_returns[mask_finite]

# Losses are negative returns (so larger positive number = worse loss)
sim_port_losses = -sim_port_returns

# VaR is the loss quantile in the tail
VaR_95 = np.quantile(sim_port_losses, 1 - alpha_95)  # 95th percentile of losses
VaR_99 = np.quantile(sim_port_losses, 1 - alpha_99)  # 99th percentile of losses

print(f"\n1-Day VaR (95%): {VaR_95:.4%}")
print(f"1-Day VaR (99%): {VaR_99:.4%}")

# CVaR / Expected Shortfall = average loss beyond VaR threshold
cvar_95 = sim_port_losses[sim_port_losses >= VaR_95].mean()
cvar_99 = sim_port_losses[sim_port_losses >= VaR_99].mean()

print(f"1-Day CVaR (95%): {cvar_95:.4%}")
print(f"1-Day CVaR (99%): {cvar_99:.4%}")

# Visualize loss distribution and VaR cutoffs
plt.figure(figsize=(10, 6))
plt.hist(sim_port_losses, bins=80, alpha=0.7)
plt.axvline(VaR_95, linestyle="--", label=f"VaR 95% = {VaR_95:.2%}")
plt.axvline(VaR_99, linestyle="--", label=f"VaR 99% = {VaR_99:.2%}")
plt.title("Simulated 1-Day Portfolio Loss Distribution")
plt.xlabel("Loss")
plt.ylabel("Frequency")
plt.legend()
plt.show()


# -------------------------------------------
# Step 6.1: Convert VaR/CVaR into Dollar Losses
# -------------------------------------------

#for any amount can be changed
portfolio_value = 10000

VaR_95_dollars = portfolio_value * VaR_95
VaR_99_dollars = portfolio_value * VaR_99

CVaR_95_dollars = portfolio_value * cvar_95
CVaR_99_dollars = portfolio_value * cvar_99

print("\n--- Dollar Risk Estimates ---")
print(f"Portfolio Value: ${portfolio_value:,.0f}")

print(f"1-Day VaR (95%):  ${VaR_95_dollars:,.2f}")
print(f"1-Day VaR (99%):  ${VaR_99_dollars:,.2f}")

print(f"1-Day CVaR (95%): ${CVaR_95_dollars:,.2f}")
print(f"1-Day CVaR (99%): ${CVaR_99_dollars:,.2f}")


 #Historical VaR
# Compute historical portfolio daily returns safely (float64 + dot)
w_hist = np.asarray(max_sharpe_weights, dtype=np.float64).reshape(-1)
R_hist = returns.to_numpy(dtype=np.float64)

hist_portfolio_returns = R_hist.dot(w_hist)

# Safety: keep only finite values (guards against any rare NaN/inf)
hist_portfolio_returns = hist_portfolio_returns[np.isfinite(hist_portfolio_returns)]
hist_portfolio_losses = -hist_portfolio_returns

hist_VaR_95 = np.quantile(hist_portfolio_losses, 0.95)
hist_VaR_99 = np.quantile(hist_portfolio_losses, 0.99)

print("\n=== Historical VaR ===")
print(f"Historical 1-Day VaR (95%): {hist_VaR_95:.4%}")
print(f"Historical 1-Day VaR (99%): {hist_VaR_99:.4%}")

hist_CVaR_95 = hist_portfolio_losses[hist_portfolio_losses >= hist_VaR_95].mean()
hist_CVaR_99 = hist_portfolio_losses[hist_portfolio_losses >= hist_VaR_99].mean()

print(f"Historical 1-Day CVaR (95%): {hist_CVaR_95:.4%}")
print(f"Historical 1-Day CVaR (99%): {hist_CVaR_99:.4%}")

hist_VaR_95_dollars = portfolio_value * hist_VaR_95
hist_VaR_99_dollars = portfolio_value * hist_VaR_99

print("\n--- Historical Dollar VaR ---")
print(f"Historical VaR 95%: ${hist_VaR_95_dollars:,.2f}")
print(f"Historical VaR 99%: ${hist_VaR_99_dollars:,.2f}")

plt.figure(figsize=(10, 6))
plt.hist(hist_portfolio_losses, bins=80, alpha=0.7)
plt.axvline(x=hist_VaR_95, linestyle="--", label=f"Historical VaR 95%")
plt.axvline(x=hist_VaR_99, linestyle="--", label=f"Historical VaR 99%")
plt.title("Historical 1-Day Portfolio Loss Distribution")
plt.xlabel("Loss")
plt.ylabel("Frequency")
plt.legend()
plt.show()


