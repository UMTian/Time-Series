## Business Problem

Many retail and operations teams need reliable short- to mid-term forecasts (10–100 periods) for sales, demand, or traffic to optimize inventory, staffing, and budgeting. Traditional forecasts often suffer from one of two issues:
- Too flat or under‑varied predictions that ignore real‑world volatility
- Unstable models that overshoot realistic bounds

This project provides a production‑ready Streamlit app that delivers accurate, bounded, and business‑friendly forecasts across multiple datasets with minimal setup.

## Solution Overview

The solution combines traditional statistical methods (Holt‑Winters, ARIMA) with optimized deep learning (PyTorch LSTM with attention and residual connections). It includes:
- Automatic model selection from a library of pre‑trained, quality‑scored LSTM models
- Variation‑aware output controls to avoid “flat” predictions
- Bounds validation to keep forecasts realistic and positive when needed
- A clear, interactive UI built with Streamlit to load data, select methods, and visualize results

## How It Works

1) Data Processing
- Smart numeric column detection (includes numeric‑like strings)
- Optional cleaning for domain fields (e.g., replacing zero ratings, forward/back fill)
- Train/test split utilities and seasonal decomposition

2) Forecasting Methods
- Traditional: Holt‑Winters and ARIMA with metrics (MSE, MAE, RMSE, MAPE)
- Deep Learning: Optimized LSTM models with attention, residuals, pooling, and calibrated variation mechanisms
- Prophet (optional): Daily frequency workflow with confidence intervals

3) Variation & Bounds Controls
- Multiple input/hidden/attention‑based variation sources calibrated for realistic spread
- Post‑prediction clipping using historical data range to avoid unreasonable values

4) Model Management
- Loads pre‑trained models from `optimized_models/` with configs, metrics, and scalers
- Automatically validates loaded models for output variation and quality
- Highlights best model by quality score

## Datasets Supported
- Airline Passengers (monthly)
- Daily Female Births (daily)
- Restaurant Visitors (daily)
- Superstore Sales (tabular retail time series)

## Results & Benefits
- Business‑plausible forecasts with appropriate variation
- Consistent, bounded outputs (no negatives where not allowed)
- Side‑by‑side comparison across methods
- Clear visualizations and metrics for decision‑making

## What Was Achieved
- Stable, repeatable forecasts across diverse datasets
- Fast interactive UI for analysts and non‑technical users
- Production‑oriented defaults (CPU‑only PyTorch option, pinned dependencies)

## Validation
- Automated checks: `auto_forecast_test.py`, `data_verification_test.py`, `test_fixed_forecast.py`
- In‑app validation of variation ratio and bounds

## Architecture at a Glance
- Streamlit UI (`streamlit_app.py`)
- Data & traditional models (`data_processor.py`, `traditional_forecasting.py`)
- Deep learning (`OptimizedLSTM` in `streamlit_app.py`, model files in `optimized_models/`)
- Optional Prophet (`prophet_forecasting.py`)


