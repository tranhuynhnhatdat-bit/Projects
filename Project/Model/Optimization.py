import numpy as np
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from Helper.helper import compute_filtered_sharpe
import pandas as pd
import optuna


def objective_logistic(trial, train_inputs, train_targets, test_inputs, trades_pnl):
    model = LogisticRegression(
        C=trial.suggest_float("C", 0.01, 5.0),
        max_iter=trial.suggest_int("max_iter", 500, 2000),
        class_weight="balanced",
        random_state=42,
    )
    model.fit(train_inputs, train_targets.values.ravel())
    prob_threshold = trial.suggest_float("prob_threshold", 0.3, 0.90, step=0.05)
    (
        train_sharpe,
        test_sharpe,
        base_train_sharpe,
        base_test_sharpe,
        num_trades,
    ) = compute_filtered_sharpe(
        model, train_inputs, test_inputs, trades_pnl, prob_threshold=prob_threshold
    )
    if num_trades > (len(train_inputs) + len(test_inputs)) * 0.6:
        trial.set_user_attr("model_name", "LogisticRegression")
        trial.set_user_attr("train_sharpe", train_sharpe)
        trial.set_user_attr("test_sharpe", test_sharpe)
        trial.set_user_attr("num_trades", num_trades)
        trial.set_user_attr("prob_thres", prob_threshold)
        trial.set_user_attr("model", model)
    else:
        train_sharpe = -np.inf
    return train_sharpe


def objective_random_forest(
    trial, train_inputs, train_targets, test_inputs, trades_pnl
):
    model = RandomForestClassifier(
        n_estimators=trial.suggest_int("n_estimators", 200, 1000, step=100),
        max_depth=trial.suggest_int("max_depth", 2, 8),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 5, 20),
        min_samples_split=trial.suggest_int("min_samples_split", 5, 30),
        max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        class_weight="balanced",
        random_state=42,
    )
    model.fit(train_inputs, train_targets.values.ravel())
    prob_threshold = trial.suggest_float("prob_threshold", 0.3, 0.90, step=0.05)
    (
        train_sharpe,
        test_sharpe,
        base_train_sharpe,
        base_test_sharpe,
        num_trades,
    ) = compute_filtered_sharpe(
        model, train_inputs, test_inputs, trades_pnl, prob_threshold=prob_threshold
    )
    if num_trades > (len(train_inputs) + len(test_inputs)) * 0.6:
        trial.set_user_attr("model_name", "RandomForest")
        trial.set_user_attr("train_sharpe", train_sharpe)
        trial.set_user_attr("test_sharpe", test_sharpe)
        trial.set_user_attr("num_trades", num_trades)
        trial.set_user_attr("prob_thres", prob_threshold)
        trial.set_user_attr("model", model)
    else:
        train_sharpe = -np.inf
    return train_sharpe


def objective_xgboost(trial, train_inputs, train_targets, test_inputs, trades_pnl):
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
    model.fit(train_inputs, train_targets.values.ravel())
    (
        train_sharpe,
        test_sharpe,
        base_train_sharpe,
        base_test_sharpe,
        num_trades,
    ) = compute_filtered_sharpe(
        model, train_inputs, test_inputs, trades_pnl, prob_threshold=prob_threshold
    )
    if num_trades > (len(train_inputs) + len(test_inputs)) * 0.6:
        trial.set_user_attr("model_name", "XGBoost")
        trial.set_user_attr("train_sharpe", train_sharpe)
        trial.set_user_attr("test_sharpe", test_sharpe)
        trial.set_user_attr("num_trades", num_trades)
        trial.set_user_attr("prob_thres", prob_threshold)
        trial.set_user_attr("model", model)
    else:
        train_sharpe = -np.inf
    return train_sharpe


objective_functions = {
    "LogisticRegression": objective_logistic,
    "RandomForest": objective_random_forest,
    "XGBoost": objective_xgboost,
}


def optimize_models(
    train_inputs, train_targets, test_inputs, test_targets, trades_pnl, n_trials=100
):
    results = []
    for model_name, objective in objective_functions.items():
        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: objective(
                trial, train_inputs, train_targets, test_inputs, trades_pnl
            ),
            n_trials=n_trials,
            n_jobs=-1,
        )
        best_trial = study.best_trial
        required_keys = [
            "train_sharpe",
            "test_sharpe",
            "num_trades",
            "model",
            "prob_thres",
        ]
        for trial in study.trials:
            if all(key in trial.user_attrs for key in required_keys):
                results.append(
                    {
                        "model_name": model_name,
                        "train_sharpe": trial.user_attrs["train_sharpe"],
                        "test_sharpe": trial.user_attrs["test_sharpe"],
                        "Num_trades": trial.user_attrs["num_trades"],
                        "Prob_thres": trial.user_attrs["prob_thres"],
                        "model": trial.user_attrs["model"],
                        "best_params": best_trial.params,
                    }
                )
            else:
                continue
    results_df = pd.DataFrame(results)
    return results_df.sort_values(by="test_sharpe", ascending=False).reset_index(
        drop=True
    )
