from Metrics import Asset_Metrics
import pandas as pd
import numpy as np
import MetaTrader5 as mt5
from datetime import datetime
from GettingData import getting_data_mt5, getting_M1_data

rates = getting_data_mt5("XAUUSD", mt5.TIMEFRAME_H1)

df_M1, df_1h = getting_M1_data(
    r"C:\Users\Mr.Dat\Desktop\Projects\Data\2026.6.30XAUUSD_ftmo-M1-Forex_245.csv", "1h"
)
