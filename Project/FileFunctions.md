# Project File & Function Reference

## Overview

This document describes every Python file in the `Project/` directory, its purpose, and every function defined within it.

---

## `main.py` — Main Pipeline

**Purpose:** Orchestrates the entire workflow: load data → backtest → walk-forward optimization → final optimization → results.

### Functions

| Function | Parameters | Returns | Description |
|---|---|---|---|
| `load_and_prepare_data(csv_path)` | `csv_path`: path to M1 CSV file | `(df_m1, df_1h, df_D)` | Loads raw M1 data and returns M1, 1-hour, and daily DataFrames. |
| `run_backtest_and_clean(df_m1, df_1h)` | `df_m1`: M1 DataFrame, `df_1h`: 1h DataFrame | `(stats, trades, df_m1, df_1h, labeled)` | Runs the base backtest, then merges trade outcomes onto the data. |
| `prepare_inputs_targets(labeled, trades)` | `labeled`: feature table, `trades`: trade DataFrame | `(inputs, targets, trades_pnl)` | Splits labeled data into features (X), target (y), and per-trade PnL. |
| `run_walk_forward_optimization(inputs, targets, trades_pnl, train_size, n_trials)` | features, targets, PnL, train fraction, optuna trials | `fold_results_df` | Runs walk-forward cross-validation with model optimization on each fold. |
| `summarize_fold_results(fold_results_df)` | DataFrame of fold results | `(summary, best_model)` | Groups fold results by model name, computes mean Sharpe, returns the best model name. |
| `scale_inputs(inputs)` | `inputs`: feature DataFrame | `(scaled, scaler)` | Fits a MinMaxScaler on all inputs and returns scaled data + scaler. |
| `run_final_optimization(best_model, inputs_scaled, targets, trades_pnl, n_trials)` | best model name, scaled inputs, targets, PnL, trials | `(results_df, final_model, prob_thres, sharpe)` | Optimizes the best model on the full dataset. |
| `compute_equity_curves(final_model, inputs_scaled, prob_thres, trades_pnl, df_D, initial_balance)` | trained model, scaled data, threshold, PnL, daily data, balance | `(equity, model_equity, buy_hold, model_pnl)` | Computes equity curves for filtered strategy, base strategy, and buy-and-hold. |
| `plot_results(model_equity, equity, buy_hold)` | three equity Series | None (saves PNG) | Plots filtered vs base equity and buy-and-hold equity. |
| `print_final_results(stats, initial_balance, model_equity, model_sharpe, model_pnl, results_df)` | backtest stats, balance, equity, Sharpe, PnL, results | `summary_results` (dict) | Prints and returns final performance metrics. |

---

## `Backtester/backtester.py` — Backtesting Engine

**Purpose:** Wraps the `backtesting` library to run a signal-based strategy on M1 data.

### Functions

| Function | Parameters | Returns | Description |
|---|---|---|---|
| `backtest(df, run_backtest)` | `df`: M1 DataFrame with "Signal" column, `run_backtest`: bool flag | `stats` (Backtest stats object) | Runs a backtest: buys at Signal=1, closes at Signal=-1. Returns performance statistics. |

---

## `Data/analyzeM1.py` — Data Cleaning & Feature Engineering

**Purpose:** Loads raw M1 CSV, generates trade signals, builds hourly/daily features, and merges backtest outcomes into an ML-ready labeled table.

### Classes

| Class | Fields | Description |
|---|---|---|
| `SignalConfig` | `required_times`, `buy_time`, `sell_time` | Configuration dataclass for the default seasonality-based signal function. |

### Functions

| Function | Parameters | Returns | Description |
|---|---|---|---|
| `default_signal_func(df_m1, config)` | `df_m1`: M1 DataFrame, `config`: SignalConfig | `df_m1` with "Signal" column | Default seasonality signal: only trades on days where all required times exist. Buy at `buy_time`, sell at `sell_time`. |
| `default_feature_func(df_h1)` | `df_h1`: 1-hour OHLC DataFrame | `df_h1` with feature columns | Computes 1h features: Log_Return, Return1, Return5, ATR_Pct, Dist_EMA20, RSI14, Volatility20. |
| `clean_data_M1(file, signal_func, feature_func, signal_kwargs, feature_kwargs)` | CSV path + optional custom signal/feature functions | `(df_m1, df_h1, df_D)` | Main entry point: loads CSV, adds time columns, runs signal_func, resamples to 1h/D, runs feature_func. |
| `cleaning_backtest(stats, df_m1, df_1h, drop_columns)` | backtest stats, M1/1h DataFrames, optional drop columns | `(trades, df_m1, df_1h, labeled)` | Merges trade outcomes onto M1 data, joins 1h features, drops NaN rows, returns ML-ready labeled table. |

---

## `Data/analyze1h.py` — Intraday Seasonality Analysis

**Purpose:** Analyzes intraday return patterns (hourly, weekday, session effects) with statistical tests and visualizations.

### Functions

| Function | Parameters | Returns | Description |
|---|---|---|---|
| `analyze_intraday_seasonality(file_path, show_info)` | `file_path`: path to 1h CSV, `show_info`: bool to display plots/tables | `df` (DataFrame with Hour/Weekday/Session columns) | Loads 1h data, classifies sessions (Asian/London/Overlap/NY), runs t-tests for seasonality, optionally plots heatmaps and bar charts. |

---

## `Helper/helper.py` — Evaluation Helpers

**Purpose:** Shared utility functions for model evaluation and Sharpe ratio computation.

### Functions

| Function | Parameters | Returns | Description |
|---|---|---|---|
| `evaluate_classification(model, train_X, train_y, test_X, test_y)` | model + train/test splits | `(train_acc, test_acc, report)` | Computes accuracy and classification report for a trained model. |
| `compute_filtered_sharpe(model, train_X, test_X, trades_df, prob_threshold)` | model, train/test features, PnL DataFrame, probability threshold | `(train_sharpe, test_sharpe, base_train_sharpe, base_test_sharpe, num_trades)` | Computes annualized Sharpe ratio for trades filtered by model probability threshold. Used during walk-forward optimization. |
| `compute_final_sharpe(model, inputs, targets, trades_df, prob_threshold)` | model, full features, targets, PnL, threshold | `(sharpe, num_trades)` | Same as `compute_filtered_sharpe` but for the final full-dataset optimization. |

---

## `Model/Optimization.py` — Walk-Forward Model Optimization

**Purpose:** Optimizes Logistic Regression, Random Forest, and XGBoost models using Optuna during walk-forward cross-validation.

### Functions

| Function | Parameters | Returns | Description |
|---|---|---|---|
| `objective_logistic(trial, train_inputs, train_targets, test_inputs, trades_pnl)` | Optuna trial + train/test data | `train_sharpe` | Optuna objective for LogisticRegression. Searches C, max_iter, prob_threshold. |
| `objective_random_forest(trial, train_inputs, train_targets, test_inputs, trades_pnl)` | Optuna trial + train/test data | `train_sharpe` | Optuna objective for RandomForestClassifier. Searches n_estimators, max_depth, min_samples_leaf/split, max_features, prob_threshold. |
| `objective_xgboost(trial, train_inputs, train_targets, test_inputs, trades_pnl)` | Optuna trial + train/test data | `train_sharpe` | Optuna objective for XGBClassifier. Searches n_estimators, max_depth, learning_rate, subsample, colsample_bytree, gamma, reg_alpha/lambda, prob_threshold. |
| `optimize_models(train_inputs, train_targets, test_inputs, test_targets, trades_pnl, n_trials)` | train/test splits, PnL, number of trials | `results_df` (sorted by test_sharpe) | Runs Optuna optimization for all three model types and returns a DataFrame of results sorted by test Sharpe. |

---

## `Model/FinalOptimization.py` — Final Model Optimization

**Purpose:** Optimizes the best model (selected from walk-forward) on the full dataset.

### Functions

| Function | Parameters | Returns | Description |
|---|---|---|---|
| `objective_logistic(trial, inputs, targets, trades_pnl)` | Optuna trial + full data | `sharpe` | Same as Optimization.py but runs on full dataset. |
| `objective_random_forest(trial, inputs, targets, trades_pnl)` | Optuna trial + full data | `sharpe` | Same as Optimization.py but runs on full dataset. |
| `objective_xgboost(trial, inputs, targets, trades_pnl)` | Optuna trial + full data | `sharpe` | Same as Optimization.py but runs on full dataset. |
| `final_optimize(best_model, inputs, targets, trades_pnl, n_trials)` | best model name, full data, trials | `results_df` | Runs Optuna for the selected best model on all data. Inputs should already be scaled. |

---

## `Validation/validation.py` — Walk-Forward Cross-Validation

**Purpose:** Generates walk-forward train/test splits with internal scaling.

### Functions

| Function | Parameters | Returns | Description |
|---|---|---|---|
| `walk_forward(inputs, targets, train_size, test_size, step_size)` | features, targets, fractional sizes | Yields `(X_train, X_test, y_train, y_test)` | Generator that produces walk-forward folds. Scales each fold internally with MinMaxScaler. |

---

## `AGENTS.md` — Coding Guidelines

**Purpose:** Contains coding style guidelines for AI agents working on this project. Not a Python file.

---

## `Results/` — Output Directory

**Purpose:** Stores timestamped subdirectories containing optimization results, plots, and saved models.

| Subdirectory | Contents |
|---|---|
| `Results/1/` | Previous run artifacts |
| `Results/2/` | Previous run artifacts |
| *(new timestamped folders)* | `final_optimization.csv`, `results.json`, `Strategy_results.png`, `{model_name}.pkl` |

---

## Pipeline Flow Summary

```
main.py
  │
  ├── clean_data_M1()          [Data/analyzeM1.py]
  │     ├── default_signal_func()   → adds "Signal" column
  │     └── default_feature_func()  → adds 1h features
  │
  ├── backtest()               [Backtester/backtester.py]
  │
  ├── cleaning_backtest()      [Data/analyzeM1.py]
  │     └── merges trades + features → "labeled" table
  │
  ├── walk_forward()           [Validation/validation.py]
  │     └── yields scaled train/test folds
  │
  ├── optimize_models()        [Model/Optimization.py]
  │     └── Optuna for LR / RF / XGBoost
  │
  ├── final_optimize()         [Model/FinalOptimization.py]
  │     └── Optuna for best model on full data
  │
  └── compute_filtered_sharpe()  [Helper/helper.py]
  └── compute_final_sharpe()     [Helper/helper.py]