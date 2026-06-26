import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

gold_df = pd.read_csv(
    r"C:\Users\Mr.Dat\Desktop\Projects\Data\2026.6.25XAUUSD_ftmo-D1-Forex_245.csv",
    index_col=0,
    parse_dates=True,
)
silver_df = pd.read_csv(
    r"C:\Users\Mr.Dat\Desktop\Projects\Data\2026.6.25XAGUSD_ftmo-D1-Forex_245.csv",
    index_col=0,
    parse_dates=True,
)
df = gold_df.index
df = pd.DataFrame(df).set_index("DateTime")
df["Gold_Price"] = gold_df["Close"]
df["Silver_Price"] = silver_df["Close"]
df = df.dropna()
corr = df["Gold_Price"].corr(df["Silver_Price"])
print(f"The correlation between Gold and Silver is: {corr}")

slope, intercept, r_value, p_value, std_error = stats.linregress(
    df["Gold_Price"], df["Silver_Price"]
)
print(f"Silver = {intercept:.3f} + {slope:.3f}*Gold")
print(f"R squared is {r_value**2:.3f}")
# Changes points, line, and confidence interval to the same color/alpha
sns.regplot(
    x=df["Gold_Price"],
    y=df["Silver_Price"],
    color="gold",
    scatter_kws={"alpha": 0.2},  # Sets transparency for the points
)
plt.title("Correlation between Gold and Silver")
plt.show()
