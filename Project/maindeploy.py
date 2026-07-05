import joblib
import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime
from pathlib import Path
from datetime import datetime
from Data.analyzeM1 import clean_data_mt5_m1, cleaning_backtest

# FTMO account
login = 1513833566
password = "Nhatdat08@"
server = "FTMO-Demo"
mt5.initialize()
mt5.login(login, password, server)
# Extract financial data
symbol = "XAUUSD"
rates = mt5.copy_rates_from(symbol, mt5.TIMEFRAME_M1, datetime.now(), 99999)
df = pd.DataFrame(rates)
df.time = pd.to_datetime(df.time, unit="s")
df.columns = df.columns.str.capitalize()
df = df.rename(columns={"Time": "Datetime"})
df = df[["Datetime", "Open", "High", "Low", "Close"]]

save_dir = Path("C:\\Users\\Mr.Dat\\Desktop\\Projects\\Data")
file_name = f"{symbol}_M1_{datetime.now().strftime('%Y%m%d')}"
file_path = save_dir / file_name
if not file_path.exists():
    df.to_csv(file_path, index=False)
# Get our model
model = joblib.load(
    r"C:\Users\Mr.Dat\Desktop\Projects\Project\Results\first\XGBoost.pkl"
)
best_model = model["model"]
scaler = model["scaler"]


df_m1, df_1h, df_D = clean_data_mt5_m1(file_path)
