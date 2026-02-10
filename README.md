# Quantitative Portfolio Optimization & Risk Engine

## 📌 Project Overview
This project is a Python-based quantitative finance tool designed to construct optimal portfolios using **Modern Portfolio Theory (MPT)** and assess tail risk through **Monte Carlo Simulations**.

It automates the end-to-end workflow of financial engineering: from fetching raw market data via `yfinance`, to cleaning ticks, performing mean-variance optimization, and finally calculating **Value at Risk (VaR)** and **Conditional VaR (CVaR/Expected Shortfall)**.

**Target Use Case:** Asset Allocation, Risk Management, and Algorithmic Trading Backtesting.

## 🚀 Key Features

### 1. Modern Portfolio Theory (MPT) Implementation
- **Efficient Frontier:** Simulates 10,000+ portfolio combinations to visualize the risk-return trade-off.
- **Sharpe Ratio Optimization:** Automatically identifies the "Tangency Portfolio" (Max Sharpe) and the "Global Minimum Variance Portfolio."

### 2. Advanced Risk Metrics
- **Parametric VaR (95% & 99%):** Uses a Multivariate Normal Distribution to model potential future losses.
- **Historical VaR:** Calculates risk based on actual historical drawdowns (non-parametric).
- **Conditional VaR (CVaR):** Computes "Expected Shortfall" to quantify the average loss *beyond* the VaR threshold (capturing tail risk).

### 3. Robust Numerical Computing
- **Covariance Regularization:** Implements "Jitter" (identity matrix $\epsilon$ injection) to ensure the Covariance Matrix remains Positive Semi-Definite for Cholesky decomposition.
- **Data Sanitation Pipeline:** Automatically handles missing data, thin trading days, and infinite returns to prevent `NaN` propagation in matrix multiplication.

---

## 🛠️ Tech Stack & Mathematical Logic

| Component | Library | Mathematical Application |
| :--- | :--- | :--- |
| **Data Ingestion** | `yfinance` | Adjusted Close price retrieval & datetime indexing. |
| **Linear Algebra** | `numpy` | Matrix multiplication (`@`), Cholesky decomposition, Dot products. |
| **Data Processing** | `pandas` | Time-series handling, pct_change calculations, data cleaning. |
| **Visualization** | `matplotlib` | Efficient Frontier scatter plots, Loss Distribution histograms. |

### Mathematical Formulae Implemented

**1. Portfolio Variance (Matrix Form):**
$$\sigma_p^2 = w^T \cdot \Sigma \cdot w$$
*Where $w$ is the weight vector and $\Sigma$ is the Covariance Matrix.*

**2. Covariance Regularization (Jitter):**
$$\Sigma_{stable} = \Sigma + \epsilon \cdot I$$
*Ensures numerical stability for multivariate simulations.*

---

