import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix
import joblib

df_gold = pd.read_csv(
    r"C:\Users\Mr.Dat\Desktop\Projects\Data\2026.6.25XAUUSD_ftmo-D1-Forex_245.csv",
    index_col=0,
    parse_dates=True,
)

df_gold["Daily_Return"] = df_gold["Close"].pct_change()
df_gold = df_gold.dropna()
df_gold["Weekday"] = df_gold.index.day_name()


df_silver = pd.read_csv(
    r"C:\Users\Mr.Dat\Desktop\Projects\Data\2026.6.25XAGUSD_ftmo-D1-Forex_245.csv",
    index_col=0,
    parse_dates=True,
)

df_silver["Daily_Return"] = df_silver["Close"].pct_change()
df_silver = df_silver.dropna()
df_silver["Weekday"] = df_silver.index.day_name()
df_silver["Close_Gold"] = df_gold[["Close"]]
df_silver["Vol_20"] = df_silver["Daily_Return"].rolling(20).std()
df_silver["Gold_Return"] = df_gold["Daily_Return"]
df_silver["Profitable"] = np.where(df_silver["Daily_Return"].shift(-1) > 0, 1, 0)
df_silver = df_silver.dropna()
inputs = df_silver[["Gold_Return", "Weekday", "Vol_20", "Close_Gold"]]
targets = df_silver["Profitable"]


# Splitting Data(Training Set and Testing Set)
x_train, x_test, y_train, y_test = train_test_split(
    inputs, targets, test_size=0.4, shuffle=False
)
inputs_train = x_train.copy()
inputs_test = x_test.copy()
targets_train = y_train.copy()
targets_test = y_test.copy()
numeric_cols = ["Gold_Return", "Vol_20", "Close_Gold"]
categorical_cols = ["Weekday"]
# Scale numerical features to range(0,1)
scaler = MinMaxScaler()
scaler.fit(inputs_train[numeric_cols])
inputs_train[numeric_cols] = scaler.transform(inputs_train[numeric_cols])
inputs_test[numeric_cols] = scaler.transform(inputs_test[numeric_cols])
# One-hot encode categorical features
inputs_test[categorical_cols].nunique()
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
encoder.fit(inputs_train[categorical_cols])
encoder_cols = list(encoder.get_feature_names_out(categorical_cols))
inputs_train[encoder_cols] = encoder.transform(inputs_train[categorical_cols])
inputs_test[encoder_cols] = encoder.transform(inputs_test[categorical_cols])
# Training Model
model = LogisticRegression(solver="liblinear")
model.fit(inputs_train[numeric_cols + encoder_cols], targets_train)
model.coef_.tolist()
model.feature_names_in_.tolist()

weighted_df = pd.DataFrame(
    {"Feature": model.feature_names_in_, "Weight": model.coef_[0]}
)
sns.barplot(data=weighted_df, x="Weight", y="Feature")
# Selecting columns for evaluation
train_inputs = inputs_train[numeric_cols + encoder_cols]
test_inputs = inputs_test[numeric_cols + encoder_cols]
train_predictions = model.predict(train_inputs)
train_probability = model.predict_proba(test_inputs)
accuracy_score(targets_train, train_predictions)
confusion_matrix(targets_train, train_predictions, normalize="true")


def predict_and_plot(inputs, targets, name=""):
    preds = model.predict(inputs)

    accuracy = accuracy_score(targets, preds)
    probs = model.predict_proba(inputs)[:, 1]
    roc = roc_auc_score(targets, probs)
    f1 = f1_score(targets, preds)
    print("Accuracy: {:.2f}%".format(accuracy * 100))
    print(f"F1_Score: {f1 * 100:.2f}%")
    print(f"ROC_AUC Score: {roc * 100:.2f}%")
    cf = confusion_matrix(targets, preds, normalize="true")
    plt.figure()
    sns.heatmap(cf, annot=True)
    plt.xlabel("Prediction")
    plt.ylabel("Target")
    plt.title("{} Confusion Matrix".format(name))
    return preds


predict_and_plot(train_inputs, targets_train, "Train")
predict_and_plot(test_inputs, targets_test, "Test")


# Compare Our Intelligent Model with Dumb Models
def random_guess(inputs):
    return np.random.choice([0, 1], len(inputs))


def all_no(inputs):
    return np.full(len(inputs), 0)


accuracy_score(targets_test, random_guess(test_inputs))
accuracy_score(targets_test, all_no(test_inputs))
# Save Our Model
silver_model = {
    "model": model,
    "scaler": scaler,
    "encoder": encoder,
    "numeric_col": numeric_cols,
    "encoder_col": encoder_cols,
    "categorical_col": categorical_cols,
}
joblib.dump(silver_model, "Silver_Model.joblib")
