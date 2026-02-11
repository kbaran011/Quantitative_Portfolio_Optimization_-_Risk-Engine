# Quantitative Portfolio Optimization & Risk Engine

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Finance](https://img.shields.io/badge/Finance-Quantitative-green)
![Status](https://img.shields.io/badge/Status-Complete-success)

## 📌 Project Overview
This project is a Python-based quantitative finance tool designed to construct optimal portfolios using **Modern Portfolio Theory (MPT)** and assess tail risk through **Monte Carlo Simulations**.

It automates the end-to-end workflow of financial engineering: from fetching raw market data via `yfinance`, to cleaning ticks, performing mean-variance optimization, and finally calculating **Value at Risk (VaR)** and **Conditional VaR (CVaR)**.

**Target Use Case:** Asset Allocation, Risk Management, and Algorithmic Trading Backtesting.

---

## Mathematical Deep Dive

This engine implements rigorous mathematical concepts to ensure numerical stability and statistical accuracy.

### 1. Portfolio Variance (Matrix Algebra)
The core of the optimization relies on Linear Algebra to calculate portfolio volatility. We calculate the weighted variance of the portfolio using the Covariance Matrix ($\Sigma$) of asset returns:

$$\sigma_p^2 = w^T \cdot \Sigma \cdot w$$

* **$w$**: Vector of asset weights ($1 \times N$)
* **$\Sigma$**: Covariance matrix of asset returns ($N \times N$)
* **$w^T$**: Transpose of the weight vector

### 2. Numerical Stability: Covariance Regularization ("Jitter")
To simulate future price paths using a **Multivariate Normal Distribution**, we perform a Cholesky Decomposition of the covariance matrix. However, real-world data often results in matrices that are not strictly **Positive Semi-Definite** due to floating-point rounding errors.

To fix this, we apply a "Jitter" technique (Tikhonov Regularization) by injecting microscopic noise into the diagonal:

$$\Sigma_{stable} = \Sigma + \epsilon \cdot I$$

* **$\epsilon$**: $1e^{-12}$ (Small epsilon constant)
* **$I$**: Identity Matrix
* *Why?* This ensures the matrix is invertible and the simulation does not crash, without materially affecting the risk estimates.

### 3. Risk Metrics: VaR vs. CVaR
The engine calculates risk using two distinct methodologies to capture "Tail Risk":

* **Value at Risk (VaR):** The threshold loss level ($\alpha$) such that the probability of losing more than this amount is $1-\alpha$.
    $$P(L > VaR_\alpha) = 1 - \alpha$$

* **Conditional VaR (Expected Shortfall):** The average loss *conditional* on the loss exceeding the VaR threshold. This accounts for the magnitude of extreme "Black Swan" events that standard VaR ignores.
    $$CVaR_\alpha = E[L \mid L \ge VaR_\alpha]$$

---

## Key Features

### 1. Modern Portfolio Theory (MPT) Implementation
- **Efficient Frontier:** Simulates 10,000+ portfolio combinations to visualize the risk-return trade-off.
- **Sharpe Ratio Optimization:** Automatically identifies the "Tangency Portfolio" (Max Sharpe) and the "Global Minimum Variance Portfolio."

### 2. Advanced Risk Metrics
- **Parametric VaR (95% & 99%):** Uses a Multivariate Normal Distribution to model potential future losses.
- **Historical VaR:** Calculates risk based on actual historical drawdowns (non-parametric).
- **cvAR (Expected Shortfall):** Captures tail risk beyond the standard confidence intervals.

### 3. Robust Data Pipeline
- **Data Sanitation:** Automatically detects and filters infinite values (`inf`) caused by division-by-zero on thin trading days.
- **Missing Data Handling:** Strict `dropna()` protocols to ensure matrix alignment.

---

## Project Structure

```text
markowitz_optimization/
│
├── main.py              # Core logic: Data fetch -> Optimization -> VaR Calculation
├── requirements.txt     # Dependencies (numpy, pandas, yfinance, matplotlib)
└── README.md            # Documentation
