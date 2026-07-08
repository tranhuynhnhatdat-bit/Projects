import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime


def getting_data_mt5(
    symbol: str,
    timeframe: int,
) -> pd.DataFrame:
    # FTMO account
    login = 1513833566
    password = "Nhatdat08@"
    server = "FTMO-Demo"
    mt5.initialize()
    mt5.login(login, password, server)

    rates = mt5.copy_rates_from(symbol, timeframe, datetime.now(), 99999)
    rates = pd.DataFrame(rates)
    rates.columns = rates.columns.str.capitalize()
    rates.Time = pd.to_datetime(rates.Time, unit="s")
    rates = rates.rename(columns={"Time": "DateTime"})
    rates = rates.set_index("DateTime")
    rates = rates.drop(columns=["Tick_volume", "Spread", "Real_volume"])
    return rates


if __name__ == "__main__":
    rates = getting_data_mt5("XAUUSD", mt5.TIMEFRAME_H1)
