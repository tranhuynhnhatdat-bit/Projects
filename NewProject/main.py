from Metrics import Asset_Metrics
import pandas as pd
import numpy as np

df = pd.read_csv(
    r"C:\Users\Mr.Dat\Desktop\Projects\Data\2026.6.30XAUUSD_ftmo-M1-Forex_245.csv",
    header=None,
    names=["Datetime", "Open", "High", "Low", "Close"],
    index_col=0,
    parse_dates=True,
)
asset = Asset_Metrics(df, "Close")
