import talib
import pandas as pd
import numpy as np
from numba import njit
from Data.DataManager import DataManager


class IndicatorCal:
    def __init__(self, higher_df):
        self._higher_df = higher_df

    # ── Existing indicators ─────────────────────────────────────────

    def ATR(self, period=14):
        return talib.ATR(
            self._higher_df["High"],
            self._higher_df["Low"],
            self._higher_df["Close"],
            period,
        )

    def ADX(self, period=14):
        return talib.ADX(
            self._higher_df["High"],
            self._higher_df["Low"],
            self._higher_df["Close"],
            period,
        )

    def ADXR(self, period=14):
        return talib.ADXR(
            self._higher_df["High"],
            self._higher_df["Low"],
            self._higher_df["Close"],
            period,
        )

    def AROONOSC(self, period=14):
        return talib.AROONOSC(
            self._higher_df["High"],
            self._higher_df["Low"],
            period,
        )

    def BOP(self):
        return talib.BOP(
            self._higher_df["Open"],
            self._higher_df["High"],
            self._higher_df["Low"],
            self._higher_df["Close"],
        )

    # ── Single-output indicators (return Series) ───────────────────

    def EMA(self, period=14):
        return talib.EMA(self._higher_df["Close"], period)

    def SMA(self, period=14):
        return talib.SMA(self._higher_df["Close"], period)

    def WMA(self, period=14):
        return talib.WMA(self._higher_df["Close"], period)

    def HMA(self, period=14):
        """Hull Moving Average – manual implementation with numba."""
        close = self._higher_df["Close"].values
        return pd.Series(
            self._hull_moving_average(close, period), index=self._higher_df.index
        )

    @staticmethod
    @njit
    def _hull_moving_average(close, period):
        n = len(close)
        half = int(period / 2)
        sqrt_period = int(np.sqrt(period))

        # WMA over half period
        wma_half = np.full(n, np.nan)
        for i in range(half - 1, n):
            weights = np.arange(1, half + 1, dtype=np.float64)
            wma_half[i] = np.sum(close[i - half + 1 : i + 1] * weights) / weights.sum()

        # WMA over full period
        wma_full = np.full(n, np.nan)
        for i in range(period - 1, n):
            weights = np.arange(1, period + 1, dtype=np.float64)
            wma_full[i] = (
                np.sum(close[i - period + 1 : i + 1] * weights) / weights.sum()
            )

        # Raw HMA = 2 * WMA(half) - WMA(full)
        raw_hma = 2.0 * wma_half - wma_full

        # Final HMA = WMA of raw HMA over sqrt(period)
        hma = np.full(n, np.nan)
        for i in range(sqrt_period - 1, n):
            weights = np.arange(1, sqrt_period + 1, dtype=np.float64)
            hma[i] = (
                np.sum(raw_hma[i - sqrt_period + 1 : i + 1] * weights) / weights.sum()
            )

        return hma

    def KAMA(self, period=14):
        return talib.KAMA(self._higher_df["Close"], period)

    def RSI(self, period=14):
        return talib.RSI(self._higher_df["Close"], period)

    def CCI(self, period=14):
        return talib.CCI(
            self._higher_df["High"],
            self._higher_df["Low"],
            self._higher_df["Close"],
            period,
        )

    def ROC(self, period=14):
        return talib.ROC(self._higher_df["Close"], period)

    def Momentum(self, period=14):
        return talib.MOM(self._higher_df["Close"], period)

    def StdDev(self, period=14):
        return talib.STDDEV(self._higher_df["Close"], period)

    # ── Multi-output indicators (return new DataFrame without OHLC) ──

    def Supertrend(self, period=14, multiplier=3.0):
        """Return a new DataFrame with Supertrend_Direction and Supertrend_Signal columns."""
        high = self._higher_df["High"].values
        low = self._higher_df["Low"].values
        close = self._higher_df["Close"].values

        atr = talib.ATR(
            self._higher_df["High"],
            self._higher_df["Low"],
            self._higher_df["Close"],
            period,
        ).values

        direction, signal = self._supertrend_calc(
            high, low, close, atr, period, multiplier
        )

        result_df = self._higher_df.copy().drop(
            columns=["Open", "High", "Low", "Close"]
        )
        result_df[f"Supertrend_Direction_{period}"] = direction
        result_df[f"Supertrend_Signal_{period}"] = signal
        return result_df

    @staticmethod
    @njit
    def _supertrend_calc(high, low, close, atr, period, multiplier):
        n = len(close)
        hl_avg = (high + low) / 2.0

        basic_upper = hl_avg + multiplier * atr
        basic_lower = hl_avg - multiplier * atr

        final_upper = np.full(n, np.nan)
        final_lower = np.full(n, np.nan)
        direction = np.full(n, np.nan)
        signal = np.full(n, np.nan)

        for i in range(1, n):
            if np.isnan(basic_upper[i]) or np.isnan(basic_lower[i]):
                continue

            if np.isnan(final_upper[i - 1]):
                final_upper[i] = basic_upper[i]
            else:
                prev_up = not np.isnan(direction[i - 1]) and direction[i - 1] == 1
                if basic_upper[i] < final_upper[i - 1] and prev_up:
                    final_upper[i] = basic_upper[i]
                else:
                    final_upper[i] = final_upper[i - 1]

            if np.isnan(final_lower[i - 1]):
                final_lower[i] = basic_lower[i]
            else:
                prev_down = not np.isnan(direction[i - 1]) and direction[i - 1] == -1
                if basic_lower[i] > final_lower[i - 1] and prev_down:
                    final_lower[i] = basic_lower[i]
                else:
                    final_lower[i] = final_lower[i - 1]

            if np.isnan(direction[i - 1]):
                if close[i] > final_upper[i]:
                    direction[i] = 1
                elif close[i] < final_lower[i]:
                    direction[i] = -1
                else:
                    direction[i] = 0
            else:
                if direction[i - 1] == 1:
                    direction[i] = -1 if close[i] < final_lower[i] else 1
                else:
                    direction[i] = 1 if close[i] > final_upper[i] else -1

            signal[i] = final_lower[i] if direction[i] == 1 else final_upper[i]

        return direction, signal

    def Ichimoku(self, tenkan=9, kijun=26, senkou=52):
        """Return a new DataFrame with Ichimoku cloud columns."""
        high = self._higher_df["High"].values
        low = self._higher_df["Low"].values
        close = self._higher_df["Close"].values

        tenkan_line, kijun_line, senkou_a, senkou_b, chikou = self._ichimoku_calc(
            high, low, close, tenkan, kijun, senkou
        )

        result_df = self._higher_df.copy().drop(
            columns=["Open", "High", "Low", "Close"]
        )
        result_df["Ichimoku_Tenkan"] = tenkan_line
        result_df["Ichimoku_Kijun"] = kijun_line
        result_df["Ichimoku_SenkouA"] = senkou_a
        result_df["Ichimoku_SenkouB"] = senkou_b
        result_df["Ichimoku_Chikou"] = chikou
        return result_df

    @staticmethod
    @njit
    def _ichimoku_calc(high, low, close, tenkan_p, kijun_p, senkou_b_p):
        n = len(close)

        tenkan = np.full(n, np.nan)
        kijun = np.full(n, np.nan)
        senkou_a = np.full(n, np.nan)
        senkou_b = np.full(n, np.nan)
        chikou = np.full(n, np.nan)

        for i in range(n):
            if i >= tenkan_p - 1:
                tenkan[i] = (
                    np.max(high[i - tenkan_p + 1 : i + 1])
                    + np.min(low[i - tenkan_p + 1 : i + 1])
                ) / 2.0

            if i >= kijun_p - 1:
                kijun[i] = (
                    np.max(high[i - kijun_p + 1 : i + 1])
                    + np.min(low[i - kijun_p + 1 : i + 1])
                ) / 2.0

        for i in range(n):
            if i >= senkou_b_p - 1:
                senkou_b[i] = (
                    np.max(high[i - senkou_b_p + 1 : i + 1])
                    + np.min(low[i - senkou_b_p + 1 : i + 1])
                ) / 2.0

        for i in range(n):
            if not np.isnan(tenkan[i]) and not np.isnan(kijun[i]):
                idx = i + kijun_p
                if idx < n:
                    senkou_a[idx] = (tenkan[i] + kijun[i]) / 2.0

            if not np.isnan(senkou_b[i]):
                idx = i + kijun_p
                if idx < n:
                    senkou_b[idx] = senkou_b[i]

        for i in range(n):
            idx = i - kijun_p
            if idx >= 0:
                chikou[idx] = close[i]

        return tenkan, kijun, senkou_a, senkou_b, chikou

    def MACD(self, fastperiod=12, slowperiod=26, signalperiod=9):
        """Return a new DataFrame with MACD line, signal line, and histogram."""
        macd, signal, hist = talib.MACD(
            self._higher_df["Close"],
            fastperiod=fastperiod,
            slowperiod=slowperiod,
            signalperiod=signalperiod,
        )
        result_df = self._higher_df.copy().drop(
            columns=["Open", "High", "Low", "Close"]
        )
        label = f"{fastperiod}_{slowperiod}_{signalperiod}"
        result_df[f"MACD_{label}"] = macd
        result_df[f"MACD_Signal_{label}"] = signal
        result_df[f"MACD_Hist_{label}"] = hist
        return result_df

    def Stochastic(self, fastk_period=5, slowk_period=3, slowd_period=3):
        """Return a new DataFrame with Stochastic %K and %D lines."""
        slowk, slowd = talib.STOCH(
            self._higher_df["High"],
            self._higher_df["Low"],
            self._higher_df["Close"],
            fastk_period=fastk_period,
            slowk_period=slowk_period,
            slowk_matype=0,
            slowd_period=slowd_period,
            slowd_matype=0,
        )
        result_df = self._higher_df.copy().drop(
            columns=["Open", "High", "Low", "Close"]
        )
        label = f"{fastk_period}_{slowk_period}_{slowd_period}"
        result_df[f"Stoch_K_{label}"] = slowk
        result_df[f"Stoch_D_{label}"] = slowd
        return result_df

    def BollingerBands(self, period=14, nbdevup=2, nbdevdn=2):
        """Return a new DataFrame with Bollinger Bands upper, middle, and lower."""
        upper, middle, lower = talib.BBANDS(
            self._higher_df["Close"],
            timeperiod=period,
            nbdevup=nbdevup,
            nbdevdn=nbdevdn,
            matype=0,
        )
        result_df = self._higher_df.copy().drop(
            columns=["Open", "High", "Low", "Close"]
        )
        result_df[f"BB_Upper_{period}"] = upper
        result_df[f"BB_Middle_{period}"] = middle
        result_df[f"BB_Lower_{period}"] = lower
        return result_df

    def KeltnerChannel(self, period=14, multiplier=2.0):
        """Return a new DataFrame with Keltner Channel upper, middle (EMA), and lower."""
        close = self._higher_df["Close"]
        high = self._higher_df["High"]
        low = self._higher_df["Low"]

        middle = talib.EMA(close, period)
        atr = talib.ATR(high, low, close, period)

        upper = middle + multiplier * atr
        lower = middle - multiplier * atr

        result_df = self._higher_df.copy().drop(
            columns=["Open", "High", "Low", "Close"]
        )
        result_df[f"KC_Upper_{period}"] = upper
        result_df[f"KC_Middle_{period}"] = middle
        result_df[f"KC_Lower_{period}"] = lower
        return result_df

    def DonchianChannel(self, period=14):
        """Return a new DataFrame with Donchian Channel upper, middle, and lower (numba)."""
        high = self._higher_df["High"].values
        low = self._higher_df["Low"].values

        upper, middle, lower = self._donchian_calc(high, low, period)

        result_df = self._higher_df.copy().drop(
            columns=["Open", "High", "Low", "Close"]
        )
        result_df[f"DC_Upper_{period}"] = upper
        result_df[f"DC_Middle_{period}"] = middle
        result_df[f"DC_Lower_{period}"] = lower
        return result_df

    @staticmethod
    @njit
    def _donchian_calc(high, low, period):
        n = len(high)
        upper = np.full(n, np.nan)
        lower = np.full(n, np.nan)
        middle = np.full(n, np.nan)

        for i in range(period - 1, n):
            upper[i] = np.max(high[i - period + 1 : i + 1])
            lower[i] = np.min(low[i - period + 1 : i + 1])
            middle[i] = (upper[i] + lower[i]) / 2.0

        return upper, middle, lower

    # ── Return, statistical indicators ─────────────────────────────

    def Return(self):
        """Add a 'Return' column (pct_change) to _higher_df in-place."""
        self._higher_df["Return"] = self._higher_df["Close"].pct_change()
        return self._higher_df["Return"]

    def ZScore(self, period=14):
        """Return rolling Z-score of Close: (close - rolling_mean) / rolling_std."""
        rolling_mean = self._higher_df["Close"].rolling(window=period).mean()
        rolling_std = self._higher_df["Close"].rolling(window=period).std()
        return (self._higher_df["Close"] - rolling_mean) / rolling_std

    def DistFromMean(self, period=14):
        """Return rolling distance from mean of Close: close - rolling_mean."""
        rolling_mean = self._higher_df["Close"].rolling(window=period).mean()
        return self._higher_df["Close"] - rolling_mean

    def LinRegSlope(self, period=14):
        """Return Linear Regression Slope over period (TA-Lib)."""
        return talib.LINEARREG_SLOPE(self._higher_df["Close"], period)

    def RollSkew(self, period=14):
        """Return rolling skew of pct_change() returns over period (numba)."""
        if "Return" not in self._higher_df.columns:
            self.Return()
        returns = self._higher_df["Return"].values
        return pd.Series(
            self._rolling_skew(returns, period), index=self._higher_df.index
        )

    @staticmethod
    @njit
    def _rolling_skew(values, period):
        n = len(values)
        result = np.full(n, np.nan)
        for i in range(period - 1, n):
            window = values[i - period + 1 : i + 1]
            count = 0
            sum_val = 0.0
            for v in window:
                if not np.isnan(v):
                    sum_val += v
                    count += 1
            if count < 2:
                continue
            mean = sum_val / count
            m2 = 0.0
            m3 = 0.0
            for v in window:
                if not np.isnan(v):
                    d = v - mean
                    m2 += d * d
                    m3 += d * d * d
            variance = m2 / count
            if variance > 0:
                std = np.sqrt(variance)
                skew = (m3 / count) / (std * std * std)
                if count > 2:
                    skew = skew * np.sqrt(count * (count - 1)) / (count - 2)
                result[i] = skew
            else:
                result[i] = 0.0
        return result

    def RollKurt(self, period=14):
        """Return rolling kurtosis of pct_change() returns over period (numba)."""
        if "Return" not in self._higher_df.columns:
            self.Return()
        returns = self._higher_df["Return"].values
        return pd.Series(
            self._rolling_kurt(returns, period), index=self._higher_df.index
        )

    @staticmethod
    @njit
    def _rolling_kurt(values, period):
        n = len(values)
        result = np.full(n, np.nan)
        for i in range(period - 1, n):
            window = values[i - period + 1 : i + 1]
            count = 0
            sum_val = 0.0
            for v in window:
                if not np.isnan(v):
                    sum_val += v
                    count += 1
            if count < 4:
                continue
            mean = sum_val / count
            m2 = 0.0
            m4 = 0.0
            for v in window:
                if not np.isnan(v):
                    d = v - mean
                    m2 += d * d
                    m4 += d * d * d * d
            variance = m2 / count
            if variance > 0:
                std = np.sqrt(variance)
                kurt = (m4 / count) / (std * std * std * std)
                kurt_excess = kurt - 3.0
                if count > 3:
                    kurt_excess = (
                        (count - 1)
                        * ((count + 1) * kurt_excess + 6)
                        / ((count - 2) * (count - 3))
                    )
                result[i] = kurt_excess
            else:
                result[i] = 0.0
        return result


indicator = IndicatorCal(higher_df)
