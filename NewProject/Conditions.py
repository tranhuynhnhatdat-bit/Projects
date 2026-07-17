import pandas as pd


class Conditions:
    def __init__(self, higher_df=None):
        self._higher_df = higher_df

    def greater_than(self, a, b):
        return a > b

    def less_than(self, a, b):
        return a < b

    def greater_or_equal(self, a, b):
        return a >= b

    def less_or_equal(self, a, b):
        return a <= b

    def equal(self, a, b):
        return a == b

    # ── Cross detection ──────────────────────────────────────────────

    def cross_above(self, a, b):

        if isinstance(a, pd.Series) and isinstance(b, pd.Series):
            prev_condition = a.shift(1) <= b.shift(1)
        elif isinstance(a, pd.Series):
            prev_condition = a.shift(1) <= b
        elif isinstance(b, pd.Series):
            prev_condition = a <= b.shift(1)
        else:
            raise TypeError("At least one of a or b must be a pandas Series.")

        return prev_condition & (a > b)

    def cross_below(self, a, b):

        if isinstance(a, pd.Series) and isinstance(b, pd.Series):
            prev_condition = a.shift(1) >= b.shift(1)
        elif isinstance(a, pd.Series):
            prev_condition = a.shift(1) >= b
        elif isinstance(b, pd.Series):
            prev_condition = a >= b.shift(1)
        else:
            raise TypeError("At least one of a or b must be a pandas Series.")

        return prev_condition & (a < b)

    # ── Rising / Falling ─────────────────────────────────────────────

    def rising(self, series, period):
        return series > series.shift(period)

    def falling(self, series, period):
        return series < series.shift(period)

    # ── Breakout detection ───────────────────────────────────────────

    def breaking_highest_high(self, series, period):
        return series > series.rolling(period).max().shift(1)

    def breaking_lowest_low(self, series, period):
        return series < series.rolling(period).min().shift(1)

    # ── Between ───────────────────────────────────────────────────────

    def between(self, series, high, low):
        return (series >= low) & (series <= high)

    # ── Consecutive up / down ─────────────────────────────────────────

    def n_consecutive_up(self, series, period):
        return (series.diff() > 0).rolling(period).sum() == period

    def n_consecutive_down(self, series, period):
        return (series.diff() < 0).rolling(period).sum() == period

    # ── Day-of-week conditions ────────────────────────────────────────

    def is_monday(self):
        return pd.Series(
            self._higher_df.index.dayofweek == 0, index=self._higher_df.index
        )

    def is_tuesday(self):
        return pd.Series(
            self._higher_df.index.dayofweek == 1, index=self._higher_df.index
        )

    def is_wednesday(self):
        return pd.Series(
            self._higher_df.index.dayofweek == 2, index=self._higher_df.index
        )

    def is_thursday(self):
        return pd.Series(
            self._higher_df.index.dayofweek == 3, index=self._higher_df.index
        )

    def is_friday(self):
        return pd.Series(
            self._higher_df.index.dayofweek == 4, index=self._higher_df.index
        )

    # ── Month conditions ──────────────────────────────────────────────

    def is_month_1(self):
        return pd.Series(self._higher_df.index.month == 1, index=self._higher_df.index)

    def is_month_2(self):
        return pd.Series(self._higher_df.index.month == 2, index=self._higher_df.index)

    def is_month_3(self):
        return pd.Series(self._higher_df.index.month == 3, index=self._higher_df.index)

    def is_month_4(self):
        return pd.Series(self._higher_df.index.month == 4, index=self._higher_df.index)

    def is_month_5(self):
        return pd.Series(self._higher_df.index.month == 5, index=self._higher_df.index)

    def is_month_6(self):
        return pd.Series(self._higher_df.index.month == 6, index=self._higher_df.index)

    def is_month_7(self):
        return pd.Series(self._higher_df.index.month == 7, index=self._higher_df.index)

    def is_month_8(self):
        return pd.Series(self._higher_df.index.month == 8, index=self._higher_df.index)

    def is_month_9(self):
        return pd.Series(self._higher_df.index.month == 9, index=self._higher_df.index)

    def is_month_10(self):
        return pd.Series(self._higher_df.index.month == 10, index=self._higher_df.index)

    def is_month_11(self):
        return pd.Series(self._higher_df.index.month == 11, index=self._higher_df.index)

    def is_month_12(self):
        return pd.Series(self._higher_df.index.month == 12, index=self._higher_df.index)
