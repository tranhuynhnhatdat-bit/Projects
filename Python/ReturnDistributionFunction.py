import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_and_clean_data(file_path):
    """
    Load and clean market data.
    """

    df = pd.read_csv(file_path, index_col=0, parse_dates=True)

    # Clean column names
    df.columns = df.columns.str.strip()

    # Remove unnecessary column
    if "Time" in df.columns:
        df.drop(columns="Time", inplace=True)

    # Daily returns
    df["Daily_Return"] = df["Close"].pct_change()

    # Remove first NaN
    df.dropna(inplace=True)

    return df


def create_higher_timeframe_returns(df):
    """
    Create weekly, monthly and quarterly return DataFrames.
    """

    # Weekly
    weekly_df = pd.DataFrame(df["Close"].resample("W-Fri").last().pct_change())
    weekly_df.rename(columns={"Close": "Weekly_Return"}, inplace=True)

    # Monthly
    month_df = pd.DataFrame(df["Close"].resample("ME").last().pct_change())
    month_df.rename(columns={"Close": "Monthly_Return"}, inplace=True)

    # Quarterly
    quarter_df = pd.DataFrame(df["Close"].resample("QE").last().pct_change())
    quarter_df.rename(columns={"Close": "Quarterly_Return"}, inplace=True)

    return weekly_df, month_df, quarter_df


def add_stats(ax, returns):
    """
    Add summary statistics to a chart.
    """

    text = (
        f"Mean: {returns.mean():.4f}\n"
        f"Std: {returns.std():.4f}\n"
        f"Skew: {returns.skew():.2f}\n"
        f"Kurtosis: {returns.kurtosis():.2f}"
    )

    ax.text(
        0.95,
        0.95,
        text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )


def plot_distribution(ax, returns, title, bins):
    """
    Plot histogram + KDE + statistics.
    """

    sns.histplot(data=returns, bins=bins, kde=True, stat="density", ax=ax)

    ax.set_title(title)

    add_stats(ax, returns)


def analyze_returns(file_path):
    """
    Main function.
    """

    # Load data
    df = load_and_clean_data(file_path)

    # Create returns
    weekly_df, month_df, quarter_df = create_higher_timeframe_returns(df)

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))

    # Daily
    plot_distribution(
        axes[0, 0], df["Daily_Return"], "Daily Return Distribution", bins=100
    )

    # Weekly
    plot_distribution(
        axes[0, 1], weekly_df["Weekly_Return"], "Weekly Return Distribution", bins=50
    )

    # Monthly
    plot_distribution(
        axes[1, 0], month_df["Monthly_Return"], "Monthly Return Distribution", bins=30
    )

    # Quarterly
    plot_distribution(
        axes[1, 1],
        quarter_df["Quarterly_Return"],
        "Quarterly Return Distribution",
        bins=20,
    )

    plt.tight_layout()
    plt.show()


analyze_returns(
    r"C:\Users\Mr.Dat\Desktop\Projects\Data\2026.6.20XAUUSD_ftmo-D1-Forex_247.csv"
)
analyze_returns(
    r"C:\Users\Mr.Dat\Desktop\Projects\Data\2026.6.19GBPUSD_ftmo-D1-GBPUSD_ftmo.csv"
)
