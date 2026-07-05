from sklearn.preprocessing import MinMaxScaler
import pandas as pd


def walk_forward(inputs, targets, train_size=0.2, test_size=0.2, step_size=0.1):
    """
    Walk-forward cross-validation splitter.

    Yields (X_train, X_test, y_train, y_test) where y_train/y_test are
    1D Series (not DataFrames) for compatibility with sklearn's .fit().
    """
    n = len(inputs)

    train_size = int(train_size * n)
    test_size = int(test_size * n)
    step_size = int(step_size * n)

    start = 0

    while start + train_size + test_size <= n:
        train_start = start
        train_end = start + train_size

        test_start = train_end
        test_end = test_start + test_size

        X_train = inputs.iloc[train_start:train_end]
        y_train = targets.iloc[train_start:train_end]

        X_test = inputs.iloc[test_start:test_end]
        y_test = targets.iloc[test_start:test_end]

        # Use the base columns from 'inputs' to ensure consistency
        base_columns = inputs.columns
        X_train = X_train[base_columns]
        X_test = X_test[base_columns]

        # Scale the data
        scaler = MinMaxScaler()
        X_train = pd.DataFrame(
            scaler.fit_transform(X_train), index=X_train.index, columns=base_columns
        )
        X_test = pd.DataFrame(
            scaler.transform(X_test), index=X_test.index, columns=base_columns
        )

        # Ensure targets are 1D Series (not DataFrames)
        if isinstance(y_train, pd.DataFrame):
            y_train = y_train.iloc[:, 0]
        if isinstance(y_test, pd.DataFrame):
            y_test = y_test.iloc[:, 0]

        yield X_train, X_test, y_train, y_test

        start += step_size
