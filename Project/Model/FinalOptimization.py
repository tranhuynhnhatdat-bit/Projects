from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from Helper.helper import compute_final_sharpe
import pandas as pd
import optuna


def objective_logistic(trial, inputs, targets, trades_pnl):
    model = LogisticRegression(
        C=trial.suggest_float("C", 0.01, 5.0),
        max_iter=trial.suggest_int("max_iter", 500, 2000),
        class_weight="balanced",
        random_state=42,
    )
    model.fit(inputs, targets.values.ravel())
    prob_threshold = trial.suggest_float("prob_threshold", 0.3, 0.90, step=0.05)
    sharpe, num_trades = compute_final_sharpe(
        model, inputs, targets, trades_pnl, prob_threshold
    )
    if num_trades > len(inputs) * 0.6:
        trial.set_user_attr("model_name", "LogisticRegression")
        trial.set_user_attr("sharpe", sharpe)
        trial.set_user_attr("num_trades", num_trades)
        trial.set_user_attr("prob_thres", prob_threshold)
        trial.set_user_attr("model", model)
    else:
        sharpe = -100
    return sharpe


def objective_random_forest(trial, inputs, targets, trades_pnl):
    model = RandomForestClassifier(
        n_estimators=trial.suggest_int("n_estimators", 200, 1000, step=100),
        max_depth=trial.suggest_int("max_depth", 2, 8),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 5, 20),
        min_samples_split=trial.suggest_int("min_samples_split", 5, 30),
        max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        class_weight="balanced",
        random_state=42,
    )
    model.fit(inputs, targets.values.ravel())
    prob_threshold = trial.suggest_float("prob_threshold", 0.3, 0.90, step=0.05)
    sharpe, num_trades = compute_final_sharpe(
        model, inputs, targets, trades_pnl, prob_threshold
    )
    if num_trades > len(inputs) * 0.6:
        trial.set_user_attr("model_name", "RandomForest")
        trial.set_user_attr("sharpe", sharpe)
        trial.set_user_attr("num_trades", num_trades)
        trial.set_user_attr("prob_thres", prob_threshold)
        trial.set_user_attr("model", model)
    else:
        sharpe = -100
    return sharpe


def objective_xgboost(trial, inputs, targets, trades_pnl):
    model = XGBClassifier(
        n_estimators=trial.suggest_int("n_estimators", 100, 1000, step=100),
        max_depth=trial.suggest_int("max_depth", 2, 10),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
        gamma=trial.suggest_float("gamma", 0.0, 5.0),
        reg_alpha=trial.suggest_float("reg_alpha", 0.0, 5.0),
        reg_lambda=trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
        random_state=42,
    )
    prob_threshold = trial.suggest_float("prob_threshold", 0.30, 0.90, step=0.05)
    model.fit(inputs, targets.values.ravel())
    sharpe, num_trades = compute_final_sharpe(
        model, inputs, targets, trades_pnl, prob_threshold
    )
    if num_trades > len(inputs) * 0.6:
        trial.set_user_attr("model_name", "XGBoost")
        trial.set_user_attr("sharpe", sharpe)
        trial.set_user_attr("num_trades", num_trades)
        trial.set_user_attr("prob_thres", prob_threshold)
        trial.set_user_attr("model", model)
    else:
        sharpe = -100
    return sharpe


objective_functions = {
    "LogisticRegression": objective_logistic,
    "RandomForest": objective_random_forest,
    "XGBoost": objective_xgboost,
}


def final_optimize(best_model, inputs, targets, trades_pnl, n_trials=100):
    """
    Optimize the selected best model over all data.

    NOTE: `inputs` should already be scaled (e.g. via MinMaxScaler)
    before being passed here -- this function does NOT scale internally
    to avoid double-scaling in the pipeline.
    """
    best_objective = objective_functions.get(best_model)
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: best_objective(trial, inputs, targets, trades_pnl),
        n_jobs=-1,
        n_trials=n_trials,
    )
    required_keys = ["model_name", "sharpe", "num_trades", "prob_thres", "model"]
    results = []
    for trial in study.trials:
        if all(key in trial.user_attrs for key in required_keys):
            results.append(
                {
                    "model_name": best_model,
                    "sharpe": trial.user_attrs["sharpe"],
                    "num_trades": trial.user_attrs["num_trades"],
                    "prob_thres": trial.user_attrs["prob_thres"],
                    "model": trial.user_attrs["model"],
                }
            )
    return pd.DataFrame(results)
