from Data.DataManager import DataManager
from Conditions import Conditions
from IndicatorsCal import IndicatorCal

from dataclasses import dataclass, field


@dataclass
class IndicatorDefinition:
    name: str
    method_name: str
    outputs: str
    parameter_ranges: dict
    category: str


@dataclass
class ConditionDefinition:
    name: str
    n_inputs: int  # number of Series required
    allow_constant: bool  # can compare to a constant?
    compatible_categories: list[str]  # which indicator categories are valid
    constant_ranges: dict[str, tuple] | None = None  # category -> (min, max)


@dataclass
class ExitDefinition:
    name: str
    parameter_ranges: dict = field(default_factory=dict)


@dataclass
class GeneratorConfig:
    # ----------------------
    # Indicators
    # ----------------------
    indicators: list[IndicatorDefinition] = field(
        default_factory=lambda: [
            # ── Moving Averages (price) ─────────────────────────────
            IndicatorDefinition(
                name="EMA",
                method_name="EMA",
                outputs=["value"],
                parameter_ranges={"period": (5, 200)},
                category="price",
            ),
            IndicatorDefinition(
                name="SMA",
                method_name="SMA",
                outputs=["value"],
                parameter_ranges={"period": (5, 200)},
                category="price",
            ),
            IndicatorDefinition(
                name="WMA",
                method_name="WMA",
                outputs=["value"],
                parameter_ranges={"period": (5, 200)},
                category="price",
            ),
            IndicatorDefinition(
                name="HMA",
                method_name="HMA",
                outputs=["value"],
                parameter_ranges={"period": (5, 200)},
                category="price",
            ),
            IndicatorDefinition(
                name="KAMA",
                method_name="KAMA",
                outputs=["value"],
                parameter_ranges={"period": (5, 50)},
                category="price",
            ),
            # ── Momentum / Oscillators ──────────────────────────────
            IndicatorDefinition(
                name="RSI",
                method_name="RSI",
                outputs=["value"],
                parameter_ranges={"period": (5, 30)},
                category="oscillator",
            ),
            IndicatorDefinition(
                name="CCI",
                method_name="CCI",
                outputs=["value"],
                parameter_ranges={"period": (5, 50)},
                category="oscillator",
            ),
            IndicatorDefinition(
                name="ROC",
                method_name="ROC",
                outputs=["value"],
                parameter_ranges={"period": (5, 50)},
                category="oscillator",
            ),
            IndicatorDefinition(
                name="Momentum",
                method_name="Momentum",
                outputs=["value"],
                parameter_ranges={"period": (5, 50)},
                category="oscillator",
            ),
            IndicatorDefinition(
                name="StochK",
                method_name="StochK",
                outputs=["value"],
                parameter_ranges={
                    "fastk_period": (3, 20),
                    "slowk_period": (2, 10),
                    "slowd_period": (2, 10),
                },
                category="oscillator",
            ),
            IndicatorDefinition(
                name="StochD",
                method_name="StochD",
                outputs=["value"],
                parameter_ranges={
                    "fastk_period": (3, 20),
                    "slowk_period": (2, 10),
                    "slowd_period": (2, 10),
                },
                category="oscillator",
            ),
            # ── Volatility ──────────────────────────────────────────
            IndicatorDefinition(
                name="ATR",
                method_name="ATR",
                outputs=["value"],
                parameter_ranges={"period": (5, 50)},
                category="volatility",
            ),
            IndicatorDefinition(
                name="StdDev",
                method_name="StdDev",
                outputs=["value"],
                parameter_ranges={"period": (5, 50)},
                category="volatility",
            ),
            # ── Trend / Direction ───────────────────────────────────
            IndicatorDefinition(
                name="ADX",
                method_name="ADX",
                outputs=["value"],
                parameter_ranges={"period": (5, 50)},
                category="trend",
            ),
            IndicatorDefinition(
                name="ADXR",
                method_name="ADXR",
                outputs=["value"],
                parameter_ranges={"period": (5, 50)},
                category="trend",
            ),
            IndicatorDefinition(
                name="AROONOSC",
                method_name="AROONOSC",
                outputs=["value"],
                parameter_ranges={"period": (5, 50)},
                category="oscillator",
            ),
            IndicatorDefinition(
                name="BOP",
                method_name="BOP",
                outputs=["value"],
                parameter_ranges={},
                category="bounded",
            ),
            # ── MACD Signal & Histogram ─────────────────────────────
            IndicatorDefinition(
                name="MACDSignal",
                method_name="MACDSignal",
                outputs=["value"],
                parameter_ranges={
                    "fastperiod": (5, 20),
                    "slowperiod": (20, 50),
                    "signalperiod": (5, 15),
                },
                category="oscillator",
            ),
            IndicatorDefinition(
                name="MACDHist",
                method_name="MACDHist",
                outputs=["value"],
                parameter_ranges={
                    "fastperiod": (5, 20),
                    "slowperiod": (20, 50),
                    "signalperiod": (5, 15),
                },
                category="oscillator",
            ),
            # ── Keltner Channel (price) ─────────────────────────────
            IndicatorDefinition(
                name="KCUpper",
                method_name="KCUpper",
                outputs=["value"],
                parameter_ranges={"period": (5, 50), "multiplier": (1.0, 5.0)},
                category="price",
            ),
            IndicatorDefinition(
                name="KCMiddle",
                method_name="KCMiddle",
                outputs=["value"],
                parameter_ranges={"period": (5, 50), "multiplier": (1.0, 5.0)},
                category="price",
            ),
            IndicatorDefinition(
                name="KCLower",
                method_name="KCLower",
                outputs=["value"],
                parameter_ranges={"period": (5, 50), "multiplier": (1.0, 5.0)},
                category="price",
            ),
            # ── Donchian Channel (price) ────────────────────────────
            IndicatorDefinition(
                name="DonchianHigh",
                method_name="DonchianHigh",
                outputs=["value"],
                parameter_ranges={"period": (5, 50)},
                category="price",
            ),
            IndicatorDefinition(
                name="DonchianMiddle",
                method_name="DonchianMiddle",
                outputs=["value"],
                parameter_ranges={"period": (5, 50)},
                category="price",
            ),
            IndicatorDefinition(
                name="DonchianLow",
                method_name="DonchianLow",
                outputs=["value"],
                parameter_ranges={"period": (5, 50)},
                category="price",
            ),
            # ── Supertrend ──────────────────────────────────────────
            IndicatorDefinition(
                name="SupertrendDirection",
                method_name="SupertrendDirection",
                outputs=["value"],
                parameter_ranges={"period": (5, 30), "multiplier": (1.0, 5.0)},
                category="bounded",
            ),
            IndicatorDefinition(
                name="SupertrendSignal",
                method_name="SupertrendSignal",
                outputs=["value"],
                parameter_ranges={"period": (5, 30), "multiplier": (1.0, 5.0)},
                category="price",
            ),
            # ── Ichimoku (price) ────────────────────────────────────
            IndicatorDefinition(
                name="IchimokuTenkan",
                method_name="IchimokuTenkan",
                outputs=["value"],
                parameter_ranges={
                    "tenkan": (5, 20),
                    "kijun": (20, 50),
                    "senkou": (40, 60),
                },
                category="price",
            ),
            IndicatorDefinition(
                name="IchimokuKijun",
                method_name="IchimokuKijun",
                outputs=["value"],
                parameter_ranges={
                    "tenkan": (5, 20),
                    "kijun": (20, 50),
                    "senkou": (40, 60),
                },
                category="price",
            ),
            IndicatorDefinition(
                name="IchimokuSenkouA",
                method_name="IchimokuSenkouA",
                outputs=["value"],
                parameter_ranges={
                    "tenkan": (5, 20),
                    "kijun": (20, 50),
                    "senkou": (40, 60),
                },
                category="price",
            ),
            IndicatorDefinition(
                name="IchimokuSenkouB",
                method_name="IchimokuSenkouB",
                outputs=["value"],
                parameter_ranges={
                    "tenkan": (5, 20),
                    "kijun": (20, 50),
                    "senkou": (40, 60),
                },
                category="price",
            ),
            IndicatorDefinition(
                name="IchimokuChikou",
                method_name="IchimokuChikou",
                outputs=["value"],
                parameter_ranges={
                    "tenkan": (5, 20),
                    "kijun": (20, 50),
                    "senkou": (40, 60),
                },
                category="price",
            ),
            # ── Return & Statistical ────────────────────────────────
            IndicatorDefinition(
                name="Return",
                method_name="Return",
                outputs=["value"],
                parameter_ranges={},
                category="statistical",
            ),
            IndicatorDefinition(
                name="ZScore",
                method_name="ZScore",
                outputs=["value"],
                parameter_ranges={"period": (5, 50)},
                category="oscillator",
            ),
            IndicatorDefinition(
                name="DistFromMean",
                method_name="DistFromMean",
                outputs=["value"],
                parameter_ranges={"period": (5, 50)},
                category="price",
            ),
            IndicatorDefinition(
                name="LinRegSlope",
                method_name="LinRegSlope",
                outputs=["value"],
                parameter_ranges={"period": (5, 50)},
                category="statistical",
            ),
            IndicatorDefinition(
                name="RollSkew",
                method_name="RollSkew",
                outputs=["value"],
                parameter_ranges={"period": (5, 50)},
                category="statistical",
            ),
            IndicatorDefinition(
                name="RollKurt",
                method_name="RollKurt",
                outputs=["value"],
                parameter_ranges={"period": (5, 50)},
                category="statistical",
            ),
        ]
    )

    # ----------------------
    # Conditions
    # ----------------------

    # Shared constant ranges used by comparison conditions
    _ALL_CATEGORIES = [
        "price",
        "oscillator",
        "trend",
        "volatility",
        "bounded",
        "statistical",
    ]
    _COMPARE_RANGES = {
        "price": (None, None),
        "oscillator": (-500, 500),
        "trend": (0, 100),
        "volatility": (0, None),
        "bounded": (-2, 2),
        "statistical": (None, None),
    }

    conditions: list[ConditionDefinition] = field(
        default_factory=lambda: [
            # ── Comparison (allow constant) ─────────────────────────
            ConditionDefinition(
                name="greater_than",
                n_inputs=2,
                allow_constant=True,
                compatible_categories=GeneratorConfig._ALL_CATEGORIES,
                constant_ranges=GeneratorConfig._COMPARE_RANGES,
            ),
            ConditionDefinition(
                name="less_than",
                n_inputs=2,
                allow_constant=True,
                compatible_categories=GeneratorConfig._ALL_CATEGORIES,
                constant_ranges=GeneratorConfig._COMPARE_RANGES,
            ),
            ConditionDefinition(
                name="greater_or_equal",
                n_inputs=2,
                allow_constant=True,
                compatible_categories=GeneratorConfig._ALL_CATEGORIES,
                constant_ranges=GeneratorConfig._COMPARE_RANGES,
            ),
            ConditionDefinition(
                name="less_or_equal",
                n_inputs=2,
                allow_constant=True,
                compatible_categories=GeneratorConfig._ALL_CATEGORIES,
                constant_ranges=GeneratorConfig._COMPARE_RANGES,
            ),
            ConditionDefinition(
                name="equal",
                n_inputs=2,
                allow_constant=True,
                compatible_categories=["price", "bounded"],
                constant_ranges={
                    "price": (None, None),
                    "bounded": (-2, 2),
                },
            ),
            # ── Cross detection (no constant) ───────────────────────
            ConditionDefinition(
                name="cross_above",
                n_inputs=2,
                allow_constant=False,
                compatible_categories=["price", "oscillator", "trend"],
                constant_ranges=None,
            ),
            ConditionDefinition(
                name="cross_below",
                n_inputs=2,
                allow_constant=False,
                compatible_categories=["price", "oscillator", "trend"],
                constant_ranges=None,
            ),
            # ── Rising / Falling (no constant) ──────────────────────
            ConditionDefinition(
                name="rising",
                n_inputs=1,
                allow_constant=False,
                compatible_categories=[
                    "price",
                    "oscillator",
                    "trend",
                    "volatility",
                    "statistical",
                ],
                constant_ranges=None,
            ),
            ConditionDefinition(
                name="falling",
                n_inputs=1,
                allow_constant=False,
                compatible_categories=[
                    "price",
                    "oscillator",
                    "trend",
                    "volatility",
                    "statistical",
                ],
                constant_ranges=None,
            ),
            # ── Breakout (no constant) ──────────────────────────────
            ConditionDefinition(
                name="breaking_highest_high",
                n_inputs=1,
                allow_constant=False,
                compatible_categories=["price"],
                constant_ranges=None,
            ),
            ConditionDefinition(
                name="breaking_lowest_low",
                n_inputs=1,
                allow_constant=False,
                compatible_categories=["price"],
                constant_ranges=None,
            ),
            # ── Between (can compare to constants) ──────────────────
            ConditionDefinition(
                name="between",
                n_inputs=3,
                allow_constant=True,
                compatible_categories=["price", "oscillator"],
                constant_ranges={
                    "price": (None, None),
                    "oscillator": (-500, 500),
                },
            ),
            # ── Consecutive (no constant) ───────────────────────────
            ConditionDefinition(
                name="n_consecutive_up",
                n_inputs=1,
                allow_constant=False,
                compatible_categories=["price", "oscillator", "trend", "volatility"],
                constant_ranges=None,
            ),
            ConditionDefinition(
                name="n_consecutive_down",
                n_inputs=1,
                allow_constant=False,
                compatible_categories=["price", "oscillator", "trend", "volatility"],
                constant_ranges=None,
            ),
            # ── Day-of-week (no inputs, no constant) ────────────────
            ConditionDefinition(
                name="is_monday",
                n_inputs=0,
                allow_constant=False,
                compatible_categories=[],
                constant_ranges=None,
            ),
            ConditionDefinition(
                name="is_tuesday",
                n_inputs=0,
                allow_constant=False,
                compatible_categories=[],
                constant_ranges=None,
            ),
            ConditionDefinition(
                name="is_wednesday",
                n_inputs=0,
                allow_constant=False,
                compatible_categories=[],
                constant_ranges=None,
            ),
            ConditionDefinition(
                name="is_thursday",
                n_inputs=0,
                allow_constant=False,
                compatible_categories=[],
                constant_ranges=None,
            ),
            ConditionDefinition(
                name="is_friday",
                n_inputs=0,
                allow_constant=False,
                compatible_categories=[],
                constant_ranges=None,
            ),
            # ── Month (no inputs, no constant) ──────────────────────
            ConditionDefinition(
                name="is_month_1",
                n_inputs=0,
                allow_constant=False,
                compatible_categories=[],
                constant_ranges=None,
            ),
            ConditionDefinition(
                name="is_month_2",
                n_inputs=0,
                allow_constant=False,
                compatible_categories=[],
                constant_ranges=None,
            ),
            ConditionDefinition(
                name="is_month_3",
                n_inputs=0,
                allow_constant=False,
                compatible_categories=[],
                constant_ranges=None,
            ),
            ConditionDefinition(
                name="is_month_4",
                n_inputs=0,
                allow_constant=False,
                compatible_categories=[],
                constant_ranges=None,
            ),
            ConditionDefinition(
                name="is_month_5",
                n_inputs=0,
                allow_constant=False,
                compatible_categories=[],
                constant_ranges=None,
            ),
            ConditionDefinition(
                name="is_month_6",
                n_inputs=0,
                allow_constant=False,
                compatible_categories=[],
                constant_ranges=None,
            ),
            ConditionDefinition(
                name="is_month_7",
                n_inputs=0,
                allow_constant=False,
                compatible_categories=[],
                constant_ranges=None,
            ),
            ConditionDefinition(
                name="is_month_8",
                n_inputs=0,
                allow_constant=False,
                compatible_categories=[],
                constant_ranges=None,
            ),
            ConditionDefinition(
                name="is_month_9",
                n_inputs=0,
                allow_constant=False,
                compatible_categories=[],
                constant_ranges=None,
            ),
            ConditionDefinition(
                name="is_month_10",
                n_inputs=0,
                allow_constant=False,
                compatible_categories=[],
                constant_ranges=None,
            ),
            ConditionDefinition(
                name="is_month_11",
                n_inputs=0,
                allow_constant=False,
                compatible_categories=[],
                constant_ranges=None,
            ),
            ConditionDefinition(
                name="is_month_12",
                n_inputs=0,
                allow_constant=False,
                compatible_categories=[],
                constant_ranges=None,
            ),
        ]
    )

    # ----------------------
    # Exit Types
    # ----------------------

    exits: list[ExitDefinition] = field(
        default_factory=lambda: [
            ExitDefinition(name="time_exit", parameter_ranges={"bars": (5, 50)}),
            ExitDefinition(
                name="atr_stop",
                parameter_ranges={"atr_period": (5, 30), "multiplier": (1.0, 5.0)},
            ),
        ]
    )

    # ----------------------
    # Strategy limits
    # ----------------------

    min_entry_conditions: int = 1
    max_entry_conditions: int = 5

    min_exit_conditions: int = 1
    max_exit_conditions: int = 5


config = GeneratorConfig()
