import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


def display_period(data):
    print(f"Start Date: {data.index[0].date()}")
    print(f"End Date: {data.index[-1].date()}")
    days = (data.index[-1] - data.index[0]).days
    num_years = days / 365
    print(f"Period: {math.floor(num_years)} years")
    return data


def calculate_metrics(data):
    # Total Return
    total_return = (data.iloc[-1, 3] / data.iloc[0, 3]) - 1
    print(f"Total Return: {total_return * 100:.2f}%")
    # Annualized Return
    days = (data.index[-1] - data.index[0]).days
    num_years = days / 365
    annualized_return = (1 + total_return) ** (1 / num_years) - 1
    print(f"Annualized Return: {annualized_return:.3f}%")
    # Volatility
    returns = data["Close"].pct_change()
    returns = returns.dropna()
    daily_volatility = returns.std()
    annualized_volatility = daily_volatility * np.sqrt(252)
    print(f"Annualized Volatility: {annualized_volatility * 100:.2f}%")
    # Max Drawdown
    running_max = data["Close"].cummax()  # remember the highest price in a serie
    drawdown = (
        data["Close"] / running_max
    ) - 1  # calculate drawdown each day from the peak
    max_drawdown = drawdown.min()
    print(f"Maximum Drawdown: {max_drawdown * 100:.2f}%")
    # Worst Day
    print(f"Worst Day: {returns.min() * 100:.2f}%")
    # Best Day
    print(f"Best Day: {returns.max() * 100:.2f}%")
    # Best Month
    monthly_return = data["Close"].resample("ME").last().pct_change()
    print(f"Best Month: {monthly_return.max() * 100:.2f}%")
    # Worst Month
    print(f"Worst Month: {monthly_return.min() * 100:.2f}%")
    # Skew
    print(f"Skew: {returns.skew():.1f}")
    return data


def visualization(data, asset):
    # CHANGED: Layout is now 3 rows, 2 columns (Size changed to 20x18)
    fig, axes = plt.subplots(3, 2, figsize=(20, 18))

    # Chart 1: Price Chart
    axes[0, 0].plot(data["Close"], label="Close Price")
    axes[0, 0].set_title(f"{asset}")
    axes[0, 0].legend()

    # Chart 2: Return Distribution
    returns = data["Close"].pct_change()
    axes[0, 1].hist(returns, bins=100, edgecolor="green")
    axes[0, 1].set_title("Returns Distribution")

    # Chart 3: Drawdown Curve
    running_max = data["Close"].cummax()
    drawdown = data["Close"] / running_max - 1
    axes[1, 0].plot(drawdown * 100, label="Drawdown Curve")
    axes[1, 0].set_title("Drawdown (%)")
    axes[1, 0].legend()

    # Chart 4: Monthly Return Heatmap
    monthly_returns = data["Close"].resample("ME").last().pct_change()
    monthly_df = monthly_returns.to_frame(name="Return")
    monthly_df["Year"] = monthly_df.index.year
    monthly_df["Month"] = monthly_df.index.month
    heatmap_data = monthly_df.pivot_table(
        index="Year", columns="Month", values="Return", aggfunc="mean"
    )
    im = axes[1, 1].imshow(heatmap_data * 100, aspect="auto", cmap="RdYlGn")
    axes[1, 1].set_title("Monthly Returns Heatmap (%)")
    axes[1, 1].set_xlabel("Month")
    axes[1, 1].set_ylabel("Year")
    axes[1, 1].set_xticks(np.arange(12))
    axes[1, 1].set_xticklabels(
        [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]
    )
    axes[1, 1].set_yticks(np.arange(len(heatmap_data.index)))
    axes[1, 1].set_yticklabels(heatmap_data.index)
    fig.colorbar(im, ax=axes[1, 1])

    # ================= NEW CODE HERE =================
    # Chart 5: Weekday Sum of Returns
    # Get the last close price of each day to calculate clean daily returns
    daily_data = data["Close"].resample("D").last().dropna()
    daily_returns = daily_data.pct_change().dropna()

    # Create a DataFrame and group by day of the week
    weekday_df = daily_returns.to_frame(name="Return")
    weekday_df["Weekday"] = weekday_df.index.day_name()

    # Order them correctly from Monday to Friday
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    weekday_sum = weekday_df.groupby("Weekday")["Return"].mean().reindex(day_order)

    # Plot the bar chart (values multiplied by 100 for percentage)
    bars = axes[2, 0].bar(
        weekday_sum.index, weekday_sum * 100, color="skyblue", edgecolor="black"
    )
    axes[2, 0].set_title("Mean of Daily Returns by Weekday (%)")
    axes[2, 0].set_ylabel("Total Return (%)")
    axes[2, 0].axhline(0, color="black", linewidth=0.8, linestyle="--")  # zero line
    # Remove the empty 6th subplot (bottom right) so it looks clean
    fig.delaxes(axes[2, 1])
    # =================================================

    plt.tight_layout()
    plt.show()


def market_analyzer(file, asset):
    print(f"Asset: {asset}")
    # Read and clean file
    df = pd.read_csv(file, index_col=0, parse_dates=True)
    df.columns = df.columns.str.strip()
    df = df.drop(columns="Time")
    display_period(df)
    calculate_metrics(df)
    visualization(df, asset)


market_analyzer("Data\\2026.6.19GBPUSD_ftmo-D1-GBPUSD_ftmo.csv", "GBP/USD")
