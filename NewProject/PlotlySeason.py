import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import ttest_1samp
from GettingData import getting_data_mt5
import MetaTrader5 as mt5


class PlotlySeason:
    def __init__(self, df_1h: pd.DataFrame):
        self._df_1h = df_1h
        self._df_D = (
            self._df_1h.resample("D")
            .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"})
            .dropna()
        )

    @property
    def return_1h(self) -> pd.DataFrame:
        df_1h = self._df_1h.copy()
        df_1h["Return"] = df_1h["Close"].pct_change()
        df_1h["Weekday"] = df_1h.index.day_name()
        df_1h = df_1h.dropna()
        return df_1h[["Return", "Weekday"]]

    @property
    def return_D(self) -> pd.DataFrame:
        df_D = self._df_D.copy()
        df_D["Return"] = df_D["Close"].pct_change()
        df_D["Weekday"] = df_D.index.day_name()
        df_D = df_D.dropna()
        return df_D[["Return", "Weekday"]]

    @property
    def AsianSession(self) -> str:
        return "Asian Session: 1->9"

    @property
    def LondonSession(self) -> str:
        return "London Session: 10->13"

    @property
    def OverlapSession(self) -> str:
        return "Overlap Session: 14->18"

    @property
    def NewYorkSession(self) -> str:
        return "NewYork Session: 19->23"

    def seasonality_stats(self, group_cols):
        df = self.return_1h.copy()

        if isinstance(group_cols, str):
            group_cols = [group_cols]

        if "Session" in group_cols and "Hour" not in df.columns:
            df["Hour"] = df.index.hour

        for col in group_cols:
            if col == "Hour" and col not in df.columns:
                df["Hour"] = df.index.hour
            elif col == "Month" and col not in df.columns:
                df["Month"] = df.index.month
            elif col == "Day" and col not in df.columns:
                df["Day"] = df.index.day
            elif col == "Session" and col not in df.columns:

                def assign_session(hour):
                    if 1 <= hour <= 9:
                        return "Asian"
                    elif 10 <= hour <= 13:
                        return "London"
                    elif 14 <= hour <= 18:
                        return "Overlap"
                    elif 19 <= hour <= 23:
                        return "New York"
                    return None

                df["Session"] = df["Hour"].apply(assign_session)
                df = df.dropna(subset=["Session"])

        if "Weekday" in df.columns:
            df = df[~df["Weekday"].isin(["Saturday", "Sunday"])]

        results = []
        for name, group in df.groupby(group_cols):
            returns = group["Return"]
            t_stat, p_value = ttest_1samp(returns, popmean=0)
            group_name = name[0] if isinstance(name, tuple) and len(name) == 1 else name
            results.append(
                {
                    "Group": group_name,
                    "Count": len(returns),
                    "Mean Return %": returns.mean() * 100,
                    "Median Return %": returns.median() * 100,
                    "T-stats": t_stat,
                    "P-value": p_value,
                    "Win Rate %": (returns >= 0).mean() * 100,
                }
            )

        stats = pd.DataFrame(results)
        significant = stats[(abs(stats["T-stats"]) > 2) & (stats["P-value"] < 0.05)]
        significant = significant.sort_values("T-stats", key=abs, ascending=False)
        return significant, stats

    def stats_hour(self, show_chart: bool = True):
        significant, stats = self.seasonality_stats("Hour")

        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Mean Return by Hour", "Median Return by Hour"),
        )

        fig.add_trace(
            go.Scatter(
                x=stats["Group"],
                y=stats["Mean Return %"],
                mode="lines+markers",
                name="Mean Return %",
            ),
            row=1,
            col=1,
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
        fig.update_xaxes(title_text="Hour", tickvals=list(range(24)), row=1, col=1)
        fig.update_yaxes(title_text="Mean Return %", row=1, col=1)

        fig.add_trace(
            go.Scatter(
                x=stats["Group"],
                y=stats["Median Return %"],
                mode="lines+markers",
                name="Median Return %",
            ),
            row=1,
            col=2,
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=2)
        fig.update_xaxes(title_text="Hour", tickvals=list(range(24)), row=1, col=2)
        fig.update_yaxes(title_text="Median Return %", row=1, col=2)

        fig.update_layout(title_text="Hourly Seasonality", showlegend=False, width=1000)
        if show_chart:
            fig.show()
        return fig, significant.head(10)

    def stats_weekday(self, show_chart: bool = True):
        significant, stats = self.seasonality_stats("Weekday")

        weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        stats["Group"] = pd.Categorical(
            stats["Group"], categories=weekday_order, ordered=True
        )
        stats = stats.sort_values("Group")

        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Mean Return by Weekday (Line)",
                "Median Return by Weekday (Line)",
                "Mean Return by Weekday (Bar)",
                "Median Return by Weekday (Bar)",
            ),
        )

        fig.add_trace(
            go.Scatter(
                x=stats["Group"],
                y=stats["Mean Return %"],
                mode="lines+markers",
                name="Mean Return %",
            ),
            row=1,
            col=1,
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
        fig.update_yaxes(title_text="Mean Return %", row=1, col=1)

        fig.add_trace(
            go.Scatter(
                x=stats["Group"],
                y=stats["Median Return %"],
                mode="lines+markers",
                name="Median Return %",
            ),
            row=1,
            col=2,
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=2)
        fig.update_yaxes(title_text="Median Return %", row=1, col=2)

        fig.add_trace(
            go.Bar(x=stats["Group"], y=stats["Mean Return %"], name="Mean Return %"),
            row=2,
            col=1,
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
        fig.update_yaxes(title_text="Mean Return %", row=2, col=1)

        fig.add_trace(
            go.Bar(
                x=stats["Group"], y=stats["Median Return %"], name="Median Return %"
            ),
            row=2,
            col=2,
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=2)
        fig.update_yaxes(title_text="Median Return %", row=2, col=2)

        fig.update_layout(title_text="Weekday Seasonality", showlegend=False, width=800)
        if show_chart:
            fig.show()
        return fig, significant.head(10)

    def stats_weekday_hour(self, show_chart: bool = True):
        significant, stats = self.seasonality_stats(["Weekday", "Hour"])

        stats[["Weekday", "Hour"]] = pd.DataFrame(
            stats["Group"].tolist(), index=stats.index
        )
        weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        pivot = stats.pivot_table(
            values="Mean Return %", index="Hour", columns="Weekday", aggfunc="first"
        )
        pivot = pivot[weekday_order]

        fig = go.Figure()
        fig.add_trace(
            go.Heatmap(
                z=pivot.values,
                x=pivot.columns,
                y=pivot.index,
                colorscale="RdYlGn",
                zmid=0,
                text=pivot.values.round(3),
                texttemplate="%{text}",
                name="Mean Return %",
            )
        )
        fig.update_layout(
            title="Mean Return % by Hour and Weekday",
            xaxis_title="Weekday",
            yaxis_title="Hour",
        )
        if show_chart:
            fig.show()
        return fig, significant.head(10)

    def stats_month(self, show_chart: bool = True):
        significant, stats = self.seasonality_stats("Month")

        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Mean Return by Month", "Median Return by Month"),
        )

        fig.add_trace(
            go.Scatter(
                x=stats["Group"],
                y=stats["Mean Return %"],
                mode="lines+markers",
                name="Mean Return %",
            ),
            row=1,
            col=1,
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
        fig.update_xaxes(title_text="Month", tickvals=list(range(1, 13)), row=1, col=1)
        fig.update_yaxes(title_text="Mean Return %", row=1, col=1)

        fig.add_trace(
            go.Scatter(
                x=stats["Group"],
                y=stats["Median Return %"],
                mode="lines+markers",
                name="Median Return %",
            ),
            row=1,
            col=2,
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=2)
        fig.update_xaxes(title_text="Month", tickvals=list(range(1, 13)), row=1, col=2)
        fig.update_yaxes(title_text="Median Return %", row=1, col=2)

        fig.update_layout(title_text="Monthly Seasonality", showlegend=False, width=900)
        if show_chart:
            fig.show()
        return fig, significant.head(10)

    def stats_month_day(self, show_chart: bool = True):
        significant, stats = self.seasonality_stats(["Month", "Day"])

        stats[["Month", "Day"]] = pd.DataFrame(
            stats["Group"].tolist(), index=stats.index
        )
        pivot = stats.pivot_table(
            values="Mean Return %", index="Day", columns="Month", aggfunc="first"
        )

        fig = go.Figure()
        fig.add_trace(
            go.Heatmap(
                z=pivot.values,
                x=pivot.columns,
                y=pivot.index,
                colorscale="RdYlGn",
                zmid=0,
                text=pivot.values.round(3),
                texttemplate="%{text}",
                name="Mean Return %",
            )
        )
        fig.update_layout(
            title="Mean Return % by Month and Day",
            xaxis_title="Month",
            yaxis_title="Day",
        )
        if show_chart:
            fig.show()
        return fig, significant.head(10)

    def stats_monthday(self, show_chart: bool = True):
        significant, stats = self.seasonality_stats("Day")

        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=(
                "Mean Return by Day of Month",
                "Median Return by Day of Month",
            ),
        )

        fig.add_trace(
            go.Scatter(
                x=stats["Group"],
                y=stats["Mean Return %"],
                mode="lines+markers",
                name="Mean Return %",
            ),
            row=1,
            col=1,
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
        fig.update_xaxes(title_text="Day", tickvals=list(range(1, 32)), row=1, col=1)
        fig.update_yaxes(title_text="Mean Return %", row=1, col=1)

        fig.add_trace(
            go.Scatter(
                x=stats["Group"],
                y=stats["Median Return %"],
                mode="lines+markers",
                name="Median Return %",
            ),
            row=1,
            col=2,
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=2)
        fig.update_xaxes(title_text="Day", tickvals=list(range(1, 32)), row=1, col=2)
        fig.update_yaxes(title_text="Median Return %", row=1, col=2)

        fig.update_layout(title_text="Day of Month Seasonality", showlegend=False)
        if show_chart:
            fig.show()
        return fig, significant.head(10)

    def stats_session(self, show_chart: bool = True):
        significant, stats = self.seasonality_stats("Session")

        session_order = ["Asian", "London", "Overlap", "New York"]
        stats["Group"] = pd.Categorical(
            stats["Group"], categories=session_order, ordered=True
        )
        stats = stats.sort_values("Group")

        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Mean Return by Session (Line)",
                "Median Return by Session (Line)",
                "Mean Return by Session (Bar)",
                "Median Return by Session (Bar)",
            ),
        )

        fig.add_trace(
            go.Scatter(
                x=stats["Group"],
                y=stats["Mean Return %"],
                mode="lines+markers",
                name="Mean Return %",
            ),
            row=1,
            col=1,
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
        fig.update_yaxes(title_text="Mean Return %", row=1, col=1)

        fig.add_trace(
            go.Scatter(
                x=stats["Group"],
                y=stats["Median Return %"],
                mode="lines+markers",
                name="Median Return %",
            ),
            row=1,
            col=2,
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=2)
        fig.update_yaxes(title_text="Median Return %", row=1, col=2)

        fig.add_trace(
            go.Bar(x=stats["Group"], y=stats["Mean Return %"], name="Mean Return %"),
            row=2,
            col=1,
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
        fig.update_yaxes(title_text="Mean Return %", row=2, col=1)

        fig.add_trace(
            go.Bar(
                x=stats["Group"], y=stats["Median Return %"], name="Median Return %"
            ),
            row=2,
            col=2,
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=2)
        fig.update_yaxes(title_text="Median Return %", row=2, col=2)

        fig.update_layout(title_text="Session Seasonality", showlegend=False)
        if show_chart:
            fig.show()
        return fig, significant.head(10)

    def stats_session_hour(self, show_chart: bool = True):
        significant, stats = self.seasonality_stats(["Session", "Hour"])

        stats[["Session", "Hour"]] = pd.DataFrame(
            stats["Group"].tolist(), index=stats.index
        )
        session_order = ["Asian", "London", "Overlap", "New York"]

        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[
                f"{s} Session - Mean Return by Hour" for s in session_order
            ],
        )

        for i, session in enumerate(session_order):
            row, col = i // 2 + 1, i % 2 + 1
            session_data = stats[stats["Session"] == session]
            fig.add_trace(
                go.Scatter(
                    x=session_data["Hour"],
                    y=session_data["Mean Return %"],
                    mode="lines+markers",
                    name=session,
                ),
                row=row,
                col=col,
            )
            fig.add_hline(y=0, line_dash="dash", line_color="gray", row=row, col=col)
            fig.update_xaxes(title_text="Hour", row=row, col=col)
            fig.update_yaxes(title_text="Mean Return %", row=row, col=col)

        fig.update_layout(
            title_text="Session Hourly Seasonality", showlegend=False, width=900
        )
        if show_chart:
            fig.show()
        return fig, significant.head(10)

    def stats_session_weekday(self, show_chart: bool = True):
        significant, stats = self.seasonality_stats(["Session", "Weekday"])

        stats[["Session", "Weekday"]] = pd.DataFrame(
            stats["Group"].tolist(), index=stats.index
        )
        session_order = ["Asian", "London", "Overlap", "New York"]
        weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        pivot = stats.pivot_table(
            values="Mean Return %", index="Weekday", columns="Session", aggfunc="first"
        )
        pivot = pivot.reindex(weekday_order)
        pivot = pivot[session_order]

        fig = go.Figure()
        fig.add_trace(
            go.Heatmap(
                z=pivot.values,
                x=pivot.columns,
                y=pivot.index,
                colorscale="RdYlGn",
                zmid=0,
                text=pivot.values.round(3),
                texttemplate="%{text}",
                name="Mean Return %",
            )
        )
        fig.update_layout(
            title="Mean Return % by Session and Weekday",
            xaxis_title="Session",
            yaxis_title="Weekday",
        )
        if show_chart:
            fig.show()
        return fig, significant.head(10)

    def stats_session_weekday_hour(self, show_chart: bool = True):
        significant, stats = self.seasonality_stats(["Session", "Weekday", "Hour"])

        stats[["Session", "Weekday", "Hour"]] = pd.DataFrame(
            stats["Group"].tolist(), index=stats.index
        )
        weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        session_order = ["Asian", "London", "Overlap", "New York"]

        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[f"{s} Session" for s in session_order],
        )

        for i, session in enumerate(session_order):
            row, col = i // 2 + 1, i % 2 + 1
            session_data = stats[stats["Session"] == session]
            pivot = session_data.pivot_table(
                values="Mean Return %", index="Hour", columns="Weekday", aggfunc="first"
            )
            pivot = pivot.reindex(columns=weekday_order)

            fig.add_trace(
                go.Heatmap(
                    z=pivot.values,
                    x=pivot.columns,
                    y=pivot.index,
                    colorscale="RdYlGn",
                    zmid=0,
                    text=pivot.values.round(3) if pivot.values.size > 0 else None,
                    texttemplate="%{text}",
                    name=session,
                ),
                row=row,
                col=col,
            )
            fig.update_xaxes(title_text="Weekday", row=row, col=col)
            fig.update_yaxes(title_text="Hour", row=row, col=col)

        fig.update_layout(
            title_text="Mean Return % by Session, Weekday, and Hour",
            height=600,
            width=1000,
        )
        if show_chart:
            fig.show()
        return fig, significant.head(10)

    def stats_weekday_hour_month(self, show_chart: bool = True):
        significant, stats = self.seasonality_stats(["Month", "Weekday", "Hour"])

        stats[["Month", "Weekday", "Hour"]] = pd.DataFrame(
            stats["Group"].tolist(), index=stats.index
        )
        weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

        fig = make_subplots(
            rows=2,
            cols=3,
            subplot_titles=weekday_order,
        )

        for i, weekday in enumerate(weekday_order):
            row, col = i // 3 + 1, i % 3 + 1
            weekday_data = stats[stats["Weekday"] == weekday]
            pivot = weekday_data.pivot_table(
                values="Mean Return %", index="Hour", columns="Month", aggfunc="first"
            )

            fig.add_trace(
                go.Heatmap(
                    z=pivot.values,
                    x=pivot.columns,
                    y=pivot.index,
                    colorscale="RdYlGn",
                    zmid=0,
                    text=pivot.values.round(3) if pivot.values.size > 0 else None,
                    texttemplate="%{text}",
                    name=weekday,
                ),
                row=row,
                col=col,
            )
            fig.update_xaxes(title_text="Month", row=row, col=col)
            fig.update_yaxes(title_text="Hour", row=row, col=col)

        # Hide the 6th subplot (row 2, col 3)
        fig.update_xaxes(visible=False, row=2, col=3)
        fig.update_yaxes(visible=False, row=2, col=3)

        fig.update_layout(
            title_text="Mean Return % by Weekday, Hour, and Month", width=1000
        )
        if show_chart:
            fig.show()
        return fig, significant.head(10)

    def stats_weekday_month(self, show_chart: bool = True):
        significant, stats = self.seasonality_stats(["Weekday", "Month"])

        stats[["Weekday", "Month"]] = pd.DataFrame(
            stats["Group"].tolist(), index=stats.index
        )
        weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        pivot = stats.pivot_table(
            values="Mean Return %", index="Month", columns="Weekday", aggfunc="first"
        )
        pivot = pivot[weekday_order]

        fig = go.Figure()
        fig.add_trace(
            go.Heatmap(
                z=pivot.values,
                x=pivot.columns,
                y=pivot.index,
                colorscale="RdYlGn",
                zmid=0,
                text=pivot.values.round(3),
                texttemplate="%{text}",
                name="Mean Return %",
            )
        )
        fig.update_layout(
            title="Mean Return % by Month and Weekday",
            xaxis_title="Weekday",
            yaxis_title="Month",
            width=500,
        )
        if show_chart:
            fig.show()
        return fig, significant.head(10)

    def stats_month_session_weekday(self, show_chart: bool = True):
        significant, stats = self.seasonality_stats(["Month", "Weekday", "Session"])

        stats[["Month", "Weekday", "Session"]] = pd.DataFrame(
            stats["Group"].tolist(), index=stats.index
        )
        weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        session_order = ["Asian", "London", "Overlap", "New York"]

        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[f"{s} Session" for s in session_order],
        )

        for i, session in enumerate(session_order):
            row, col = i // 2 + 1, i % 2 + 1
            session_data = stats[stats["Session"] == session]
            pivot = session_data.pivot_table(
                values="Mean Return %",
                index="Month",
                columns="Weekday",
                aggfunc="first",
            )
            pivot = pivot.reindex(columns=weekday_order)

            fig.add_trace(
                go.Heatmap(
                    z=pivot.values,
                    x=pivot.columns,
                    y=pivot.index,
                    colorscale="RdYlGn",
                    zmid=0,
                    text=pivot.values.round(3) if pivot.values.size > 0 else None,
                    texttemplate="%{text}",
                    name=session,
                ),
                row=row,
                col=col,
            )
            fig.update_xaxes(title_text="Weekday", row=row, col=col)
            fig.update_yaxes(title_text="Month", row=row, col=col)

        fig.update_layout(
            title_text="Mean Return % by Month, Session, and Weekday",
            width=800,
            height=600,
        )
        if show_chart:
            fig.show()
        return fig, significant.head(10)


if __name__ == "__main__":
    df = getting_data_mt5("XAUUSD", mt5.TIMEFRAME_H1)
    season = PlotlySeason(df)
