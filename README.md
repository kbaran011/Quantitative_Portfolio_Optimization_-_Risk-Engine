# Quantitative Portfolio Optimization & Risk Engine

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Finance](https://img.shields.io/badge/Finance-Quantitative-green)
![Status](https://img.shields.io/badge/Status-Complete-success)

A Python engine that constructs optimal portfolios with **Modern Portfolio Theory** and stress-tests them with **Monte Carlo VaR/CVaR** — from raw `yfinance` data to efficient frontier to tail-risk report in one run.

It automates the end-to-end workflow: fetching market data, cleaning ticks, mean-variance optimization, and calculating **Value at Risk (VaR)** and **Conditional VaR (CVaR)**.

**Target Use Case:** Asset Allocation, Risk Management, and Algorithmic Trading Backtesting.

---

## Quick Start

```bash
pip install -r requirements.txt
python main.py
```

---

## Mathematical Deep Dive

This engine implements rigorous mathematical concepts to ensure numerical stability and statistical accuracy.

### 1. Portfolio Variance (Matrix Algebra)

The core of the optimization relies on linear algebra to calculate portfolio volatility, using the covariance matrix ($\Sigma$) of asset returns:

$$\sigma_p^2 = w^T \cdot \Sigma \cdot w$$

* **$w$**: Vector of asset weights ($1 \times N$)
* **$\Sigma$**: Covariance matrix of asset returns ($N \times N$)
* **$w^T$**: Transpose of the weight vector

### 2. Numerical Stability: Covariance Regularization ("Jitter")

To simulate future price paths from a **Multivariate Normal Distribution**, the engine performs a Cholesky decomposition of the covariance matrix. Real-world data often yields matrices that are not strictly **positive semi-definite** due to floating-point rounding errors.

To fix this, a "jitter" (Tikhonov regularization) injects microscopic noise into the diagonal:

$$\Sigma_{stable} = \Sigma + \epsilon \cdot I$$

* **$\epsilon$**: $1e^{-12}$ (small epsilon constant)
* **$I$**: Identity matrix
* *Why?* This ensures the matrix is invertible and the simulation does not crash, without materially affecting the risk estimates.

### 3. Risk Metrics: VaR vs. CVaR

The engine calculates risk using two distinct methodologies to capture tail risk:

* **Value at Risk (VaR):** The threshold loss level ($\alpha$) such that the probability of losing more than this amount is $1-\alpha$.
    $$P(L > VaR_\alpha) = 1 - \alpha$$

* **Conditional VaR (Expected Shortfall):** The average loss *conditional* on the loss exceeding the VaR threshold — capturing the magnitude of extreme "black swan" events that standard VaR ignores.
    $$CVaR_\alpha = E[L \mid L \ge VaR_\alpha]$$

---

## Key Features

### 1. Modern Portfolio Theory (MPT) Implementation
- **Efficient Frontier:** Simulates 10,000+ portfolio combinations to visualize the risk-return trade-off.
- **Sharpe Ratio Optimization:** Automatically identifies the "Tangency Portfolio" (Max Sharpe) and the "Global Minimum Variance Portfolio."

### 2. Advanced Risk Metrics
- **Parametric VaR (95% & 99%):** Uses a Multivariate Normal Distribution to model potential future losses.
- **Historical VaR:** Calculates risk based on actual historical drawdowns (non-parametric).
- **CVaR (Expected Shortfall):** Captures tail risk beyond the standard confidence intervals.

### 3. Robust Data Pipeline
- **Data Sanitation:** Automatically detects and filters infinite values (`inf`) caused by division-by-zero on thin trading days.
- **Missing Data Handling:** Strict `dropna()` protocols to ensure matrix alignment.

---

## Project Structure

```text
.
├── main.py              # Core logic: data fetch -> optimization -> VaR/CVaR calculation
├── requirements.txt     # Dependencies (numpy, pandas, yfinance, matplotlib)
└── README.md            # Documentation
```

## Why I built this

I wanted to implement the math behind portfolio construction rather than just cite it — to see where textbook Markowitz breaks on real market data. The most instructive parts were the failure modes: covariance matrices that refuse Cholesky decomposition until regularized, `inf` returns from thin trading days, and how differently parametric and historical VaR behave in the tails. This is the finance foundation that later fed into larger data products like EconSight.
