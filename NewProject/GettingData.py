import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime


def getting_data_mt5(
    symbol: str,
    timeframe: int,
) -> pd.DataFrame:
    def connecting_to_mt5(login: int, password: str, server: str) -> bool:
        if not mt5.initialize():
            return False
        authorized = mt5.login(login, password, server)

        if authorized:
            return True
        else:
            return False

    # FTMO account
    login = 1513833566
    password = "Nhatdat08@"
    server = "FTMO-Demo"

    if connecting_to_mt5(login, password, server):
        rates = mt5.copy_rates_from(symbol, timeframe, datetime.now(), 99999)
        rates = pd.DataFrame(rates)
        rates.columns = rates.columns.str.capitalize()
        rates.Time = pd.to_datetime(rates.Time, unit="s")
        rates = rates.rename(columns={"Time": "DateTime"})
        rates = rates.set_index("DateTime")
        rates = rates.drop(columns=["Tick_volume", "Spread", "Real_volume"])
        return rates
    else:
        print("Failed to fetch market data from MT5")
        return None


def getting_M1_data(file, frequency: str) -> pd.DataFrame:
    df_M1: pd.DataFrame = pd.read_csv(
        file,
        header=None,
        names=["DateTime", "Open", "High", "Low", "Close"],
        index_col=0,
        parse_dates=True,
    )
    df_return: pd.DataFrame = (
        df_M1.resample(frequency)
        .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"})
        .dropna()
    )
    return df_M1, df_return


if __name__ == "__main__":
    rates = getting_data_mt5("XAUUSD", mt5.TIMEFRAME_M1)
