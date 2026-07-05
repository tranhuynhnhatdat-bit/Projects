from sklearn.metrics import accuracy_score, classification_report
import numpy as np


# ──────────────────────────────────────────────
# Evaluation helpers
# ──────────────────────────────────────────────
def evaluate_classification(model, train_X, train_y, test_X, test_y):
    train_pred = model.predict(train_X)
    test_pred = model.predict(test_X)
    train_acc = accuracy_score(train_y, train_pred)
    test_acc = accuracy_score(test_y, test_pred)
    report = classification_report(test_y, test_pred, digits=4)
    return train_acc, test_acc, report


def compute_filtered_sharpe(model, train_X, test_X, trades_df, prob_threshold=0.5):
    """
    Only take trades where model predicts profitable (prob > threshold).
    Position = 1 (long) if predicted profitable, else 0 (flat).
    """
    # Probabilities
    train_proba = model.predict_proba(train_X)[:, 1]
    test_proba = model.predict_proba(test_X)[:, 1]

    # Filter: 1 = take trade, 0 = skip
    train_take = (train_proba > prob_threshold).astype(int)
    test_take = (test_proba > prob_threshold).astype(int)

    # Align PnL by index (timestamp) — safer than positional split
    train_pnl = trades_df.loc[train_X.index, "PnL"]
    test_pnl = trades_df.loc[test_X.index, "PnL"]

    # Returns: only take trade when model says yes
    train_returns = train_pnl.values[train_take == 1]
    test_returns = test_pnl.values[test_take == 1]

    # Base returns (original strategy, all trades)
    base_train_returns = train_pnl.values
    base_test_returns = test_pnl.values

    # Number of trades
    num_trades = np.sum(train_take) + np.sum(test_take)
    train_num_trades = np.sum(train_take)
    test_num_trades = np.sum(test_take)
    # Number of years
    train_years = (train_X.index[-1] - train_X.index[0]).days / 365.25
    test_years = (test_X.index[-1] - test_X.index[0]).days / 365.25

    def annualized_train_sharpe(returns, riskfree=0.0):
        excess = returns - riskfree
        if np.std(excess) == 0:
            return np.nan
        return (np.mean(excess) / np.std(excess)) * np.sqrt(
            train_num_trades / train_years
        )

    def annualized_test_sharpe(returns, riskfree=0.0):
        excess = returns - riskfree
        if np.std(excess) == 0:
            return np.nan
        return (np.mean(excess) / np.std(excess)) * np.sqrt(
            test_num_trades / test_years
        )

    return (
        annualized_train_sharpe(train_returns),
        annualized_test_sharpe(test_returns),
        annualized_train_sharpe(base_train_returns),
        annualized_test_sharpe(base_test_returns),
        num_trades,
    )


def compute_final_sharpe(model, inputs, targets, trades_df, prob_threshold=0.5):
    probability = model.predict_proba(inputs)[:, 1]
    taken = (probability > prob_threshold).astype(int)
    pnl = trades_df.loc[inputs.index, "PnL"]
    trade_returns = pnl.values[taken == 1]
    num_trades = np.sum(taken)
    years = (inputs.index[-1] - inputs.index[0]).days / 365.25

    def annualized_sharpe(returns, riskfree=0.0):
        excess = returns - riskfree
        if np.std(excess) == 0:
            return np.nan
        return (np.mean(excess) / np.std(excess)) * np.sqrt(num_trades / years)

    return annualized_sharpe(trade_returns), num_trades
