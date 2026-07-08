# Project/main.py
"""Main pipeline: load data -> backtest -> walk-forward -> final optimization -> results."""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from Backtester.backtester import backtest
from Data.analyzeM1 import clean_data_M1, cleaning_backtest
from Model.FinalOptimization import final_optimize
from Model.Optimization import optimize_models
from Validation.validation import walk_forward

base_dir = Path(r"C:\Users\Mr.Dat\Desktop\Projects\Project\Results")


# ------------------------------------------------------------------
# Load & prepare data
# ------------------------------------------------------------------
def load_and_prepare_data(csv_path):
    """Load raw M1 CSV and return M1, 1h, daily DataFrames."""
    df_m1, df_1h, df_D = clean_data_M1(csv_path)
    return df_m1, df_1h, df_D


# ------------------------------------------------------------------
# Backtest & clean
# ------------------------------------------------------------------
def run_backtest_and_clean(df_m1, df_1h):
    """Run the base backtest and merge trade outcomes onto the data."""
    stats = backtest(df_m1, run_backtest=True)
    trades, df_m1, df_1h, labeled = cleaning_backtest(stats, df_m1, df_1h)
    return stats, trades, df_m1, df_1h, labeled


def prepare_inputs_targets(labeled, trades):
    """Split labeled data into features, target, and per-trade PnL."""
    inputs = labeled.drop(columns="Profitable")
    targets = labeled["Profitable"]
    trades_pnl = trades.set_index("EntryTime")[["PnL"]]
    return inputs, targets, trades_pnl


# ------------------------------------------------------------------
# Walk-forward optimization (uses unscaled inputs;
# walk_forward() scales each fold internally)
# ------------------------------------------------------------------
def run_walk_forward_optimization(
    inputs, targets, trades_pnl, train_size=0.4, n_trials=20
):
    fold_results = []
    for fold, (X_train, X_test, y_train, y_test) in enumerate(
        walk_forward(inputs, targets, train_size=train_size), start=1
    ):
        print(f"Fold {fold}: ")
        print(
            f"  Train: {X_train.index[0]} to {X_train.index[-1]}  |  "
            f"Test: {X_test.index[0]} to {X_test.index[-1]}"
        )
        df_evaluated_models = optimize_models(
            X_train, y_train, X_test, y_test, trades_pnl, n_trials=n_trials
        )
        fold_results.append(df_evaluated_models)

    fold_results_df = pd.concat(fold_results, ignore_index=True)
    return fold_results_df


def summarize_fold_results(fold_results_df):
    summary = fold_results_df.groupby("model_name")["test_sharpe"].agg(
        ["mean", "median", "std", "max", "min"],
    )
    best_model = summary["mean"].idxmax()
    return summary, best_model


# ------------------------------------------------------------------
# Scale ALL inputs (for final training + inference)
# ------------------------------------------------------------------
def scale_inputs(inputs):
    scaler = MinMaxScaler()
    scaled = pd.DataFrame(
        scaler.fit_transform(inputs), index=inputs.index, columns=inputs.columns
    )
    return scaled, scaler


# ------------------------------------------------------------------
# Final optimization on *scaled* data
# ------------------------------------------------------------------
def run_final_optimization(best_model, inputs_scaled, targets, trades_pnl, n_trials=50):
    results_df = final_optimize(
        best_model, inputs_scaled, targets, trades_pnl, n_trials=n_trials
    ).sort_values(by="sharpe", ascending=False)

    final_model = results_df.iloc[0]["model"]
    model_prob_thres = results_df.iloc[0]["prob_thres"]
    model_sharpe = results_df.iloc[0]["sharpe"]
    return results_df, final_model, model_prob_thres, model_sharpe


# ------------------------------------------------------------------
# Compute & plot equity curves
# ------------------------------------------------------------------
def compute_equity_curves(
    final_model, inputs_scaled, model_prob_thres, trades_pnl, df_D, initial_balance
):
    model_predict_prob = final_model.predict_proba(inputs_scaled)[:, 1]
    trades_taken = (model_predict_prob >= model_prob_thres).astype(int)
    model_pnl = trades_pnl[(trades_taken == 1)]
    equity = initial_balance + trades_pnl["PnL"].cumsum()
    model_equity = initial_balance + model_pnl["PnL"].cumsum()
    buy_hold = initial_balance * (1 + df_D["Return"]).cumprod()
    return equity, model_equity, buy_hold, model_pnl


def plot_results(model_equity, equity, buy_hold, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(model_equity, label="Filtered")
    axes[0].plot(equity, label="Base")
    axes[0].legend()
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("Equity")
    axes[0].set_title("Filtered vs Base Strategy")

    axes[1].plot(buy_hold)
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Equity")
    axes[1].set_title("Buy and Hold")
    fig.savefig(save_dir / "Strategy_results.png", dpi=300)


def print_final_results(
    stats, initial_balance, model_equity, model_sharpe, model_pnl, results_df
):
    print("-------Final Results------")
    print(f"Initial Balance: {initial_balance}$")
    print(f"Start:      {stats['Start']}")
    print(f"End:        {stats['End']}")
    print("-------Base Strategy------")
    print(f"Final Balance: {stats['Equity Final [$]']:.2f}$")
    print(f"Win Rate: {stats['Win Rate [%]']:.2f}%")
    print(f"Sharpe Ratio: {stats['Sharpe Ratio']:.2f}")
    print(f"CAGR: {stats['CAGR [%]']:.2f}%")
    print(f"Max Drawdown: {stats['Max. Drawdown [%]']:.2f}%")
    print(f"Number of Trades: {stats['# Trades']}")
    print("-------Filtered Strategy------")
    print(f"Final Balance: {model_equity.values[-1]:.2f}$")
    print(f"Sharpe Ratio: {model_sharpe:.2f}")
    profitable = (model_pnl.values > 0).sum()
    winrate = profitable / len(model_pnl) if len(model_pnl) > 0 else 0.0
    print(f"Win Rate: {winrate * 100:.2f}%")
    years = (model_equity.index[-1] - model_equity.index[0]).days / 365
    cagr = (model_equity.values[-1] / model_equity.values[0]) ** (1 / years) - 1
    print(f"CAGR: {cagr * 100:.2f}%")
    running_max = model_equity.cummax()
    drawdown = (model_equity - running_max) / running_max
    max_drawdown = drawdown.min()
    print(f"Max Drawdown: {max_drawdown * 100:.2f}%")
    print(f"Number_trades: {results_df.iloc[0]['num_trades']}")
    summary_results = {
        "Initial Balance": float(initial_balance),
        "Final Balance": float(model_equity.iloc[-1]),
        "Sharpe": float(model_sharpe),
        "Win Rate": float(winrate * 100),
        "CAGR": float(cagr * 100),
        "Max Drawdown": float(max_drawdown * 100),
        "Trades": int(results_df.iloc[0]["num_trades"]),
    }
    return summary_results


# ==================================================================
# MAIN PIPELINE
# ==================================================================

# 1. Load & prepare data
df_m1, df_1h, df_D = load_and_prepare_data(
    r"C:\Users\Mr.Dat\Desktop\Projects\Data\2026.6.30XAUUSD_ftmo-M1-Forex_245.csv"
)

# 2. Backtest & clean
stats, trades, df_m1, df_1h, labeled = run_backtest_and_clean(df_m1, df_1h)
inputs, targets, trades_pnl = prepare_inputs_targets(labeled, trades)

# Drop rows with NaN features (early rows where 1h features aren't yet "warm")
mask = inputs.notna().all(axis=1)
inputs = inputs[mask]
targets = targets[mask]
trades_pnl = trades_pnl.loc[inputs.index]

# 3. Walk-forward optimization on unscaled data
fold_results_df = run_walk_forward_optimization(
    inputs, targets, trades_pnl, train_size=0.4, n_trials=5
)
summary, best_model = summarize_fold_results(fold_results_df)

# 4. Scale inputs for final training
inputs_scaled, scaler = scale_inputs(inputs)

# 5. Final optimization on scaled data
results_df, final_model, model_prob_thres, model_sharpe = run_final_optimization(
    best_model, inputs_scaled, targets, trades_pnl, n_trials=5
)

# 6. Compute equity curves (uses scaled data for prediction)
initial_balance = 10000
equity, model_equity, buy_hold, model_pnl = compute_equity_curves(
    final_model, inputs_scaled, model_prob_thres, trades_pnl, df_D, initial_balance
)

# 7. Create output directory named after the best model (add suffix if exists)
save_dir = base_dir / best_model
suffix = 1
while save_dir.exists():
    save_dir = base_dir / f"{best_model}_{suffix}"
    suffix += 1
save_dir.mkdir(parents=True, exist_ok=True)

# 8. Plot & print results
plot_results(model_equity, equity, buy_hold, save_dir)

summary_results = print_final_results(
    stats, initial_balance, model_equity, model_sharpe, model_pnl, results_df
)

# 9. Save artifacts
results_df.to_csv(save_dir / "final_optimization.csv", index=False)
with open(save_dir / "results.json", "w") as f:
    json.dump(summary_results, f, indent=4)

model_artifact = {
    "model": final_model,
    "scaler": scaler,
    "columns": inputs_scaled.columns,
}
joblib.dump(model_artifact, save_dir / f"{best_model}.pkl")
