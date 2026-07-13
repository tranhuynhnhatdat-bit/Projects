from PlotlySeason import PlotlySeason
from GettingData import getting_data_mt5
import MetaTrader5 as mt5
from dash import Dash, html, dcc, Output, Input, State, callback
import dash_ag_grid as dag
import plotly.graph_objects as go
import pandas as pd
from Metrics import Asset_Metrics

if not mt5.initialize():
    quit()

# Example C: Get ONLY the symbols currently visible in your MT5 Market Watch window
visible_symbols = mt5.symbols_get()
market_watch_list = [s.name for s in visible_symbols if s.visible]

# Mapping of user-friendly labels to PlotlySeason method names
STATS_FUNCTIONS = {
    "Hourly Seasonality": "stats_hour",
    "Weekday Seasonality": "stats_weekday",
    "Weekday × Hour Heatmap": "stats_weekday_hour",
    "Monthly Seasonality": "stats_month",
    "Month × Day Heatmap": "stats_month_day",
    "Day of Month Seasonality": "stats_monthday",
    "Session Seasonality": "stats_session",
    "Session × Hour": "stats_session_hour",
    "Session × Weekday Heatmap": "stats_session_weekday",
    "Session × Weekday × Hour Heatmaps": "stats_session_weekday_hour",
    "Weekday × Hour × Month Heatmaps": "stats_weekday_hour_month",
    "Month × Weekday Heatmap": "stats_weekday_month",
    "Month × Session × Weekday Heatmaps": "stats_month_session_weekday",
}

app = Dash()

app.layout = html.Div(
    children=[
        dcc.Store(id="store_data_H1"),
        dcc.Store(id="store_data_1D"),
        html.H1(
            "Seasonality Pattern",
            style={
                "color": "#00FFCC",
                "textAlign": "center",
                "fontFamily": "Inter, Helvetica, Arial, sans-serif",
                "fontWeight": "700",
                "letterSpacing": "1px",
                "marginBottom": "30px",
            },
        ),
        dropdown_input := dcc.Dropdown(options=market_watch_list, value="XAUUSD"),
        html.Br(),
        graph_output := dcc.Graph(),
        range_slider_output := dcc.RangeSlider(
            id="years-slider",
        ),
        html.Div(id="asset-metrics-output", style={"marginTop": "30px"}),
        html.Hr(style={"borderColor": "#334155", "margin": "30px 0"}),
        html.H3(
            "Seasonality Pattern Analysis",
            style={"color": "#00FFCC", "marginBottom": "15px"},
        ),
        dcc.Dropdown(
            id="stats-function-dropdown",
            options=[{"label": k, "value": v} for k, v in STATS_FUNCTIONS.items()],
            multi=True,
            placeholder="Select seasonality patterns to analyze...",
            style={
                "color": "#0F172A",
                "marginBottom": "15px",
            },
        ),
        html.Button(
            "Run Seasonality Analysis",
            id="submit-stats-btn",
            n_clicks=0,
            style={
                "backgroundColor": "#00FFCC",
                "color": "#0F172A",
                "fontWeight": "bold",
                "padding": "10px 24px",
                "border": "none",
                "borderRadius": "6px",
                "cursor": "pointer",
                "fontSize": "16px",
                "marginBottom": "30px",
            },
        ),
        html.Div(id="asset-seasonality-output", style={"marginTop": "20px"}),
    ],
    style={
        "backgroundColor": "#0F172A",
        "minHeight": "100vh",
        "margin": "-8px",
        "padding": "40px 20px",
        "fontFamily": "Inter, Helvetica, Arial, sans-serif",
    },
)


# Fetching data
@callback(
    Output("store_data_H1", "data"),
    Output("store_data_1D", "data"),
    Input(dropdown_input, "value"),
)
def get_data(symbol):
    df_1h = getting_data_mt5(symbol, mt5.TIMEFRAME_H1)
    df_1D = df_1h.resample("D").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
    )
    df_1D = df_1D.dropna()
    df_1D = df_1D.reset_index()
    df_1h = df_1h.reset_index()

    return df_1h.to_dict("records"), df_1D.to_dict("records")


# Update slider bounds when data changes
@callback(
    Output("years-slider", "min"),
    Output("years-slider", "max"),
    Output("years-slider", "marks"),
    Output("years-slider", "value"),
    Input("store_data_1D", "data"),
)
def update_slider(cached_data):
    if cached_data is None:
        return 2000, 2030, {}, [2000, 2030]

    df = pd.DataFrame(cached_data)
    df["DateTime"] = pd.to_datetime(df["DateTime"])
    df["year"] = df["DateTime"].dt.year

    min_year = int(df["year"].min())
    max_year = int(df["year"].max())
    marks = {str(year): str(year) for year in range(min_year, max_year + 1)}

    return min_year, max_year, marks, [min_year, max_year]


# Update chart based on slider selection and symbol
@callback(
    Output(graph_output, "figure"),
    Input("years-slider", "value"),
    Input(dropdown_input, "value"),
    State("store_data_1D", "data"),
)
def update_chart(selected_years, symbol, cached_data):
    if cached_data is None:
        return go.Figure()

    df = pd.DataFrame(cached_data)
    df["DateTime"] = pd.to_datetime(df["DateTime"])
    df = df.set_index("DateTime")
    df["year"] = df.index.year

    year_min, year_max = selected_years
    filtered_df = df[(df["year"] >= year_min) & (df["year"] <= year_max)]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=filtered_df.index,
            y=filtered_df["Close"],
            mode="lines",
            name=f"{symbol} Close",
            line=dict(color="#02A6FF", dash="solid"),
        )
    )
    fig.update_layout(
        title=f"{symbol} Close Price",
        xaxis_title="DateTime",
        yaxis_title="Close Price",
    )
    return fig


# Update asset metrics based on slider selection
@callback(
    Output("asset-metrics-output", "children"),
    Input("years-slider", "value"),
    State("store_data_1D", "data"),
)
def update_metrics(selected_years, cached_data):
    if cached_data is None:
        return html.Div()

    df = pd.DataFrame(cached_data)
    df["DateTime"] = pd.to_datetime(df["DateTime"])
    df = df.set_index("DateTime")
    df["year"] = df.index.year

    year_min, year_max = selected_years
    filtered_df = df[(df["year"] >= year_min) & (df["year"] <= year_max)]

    if filtered_df.empty:
        return html.Div("No data for selected year range.", style={"color": "#888"})

    metrics = Asset_Metrics(filtered_df, "Close")

    metrics_data = [
        {"Metric": "Sharpe Ratio", "Value": f"{metrics.sharpe_ratio():.2f}"},
        {"Metric": "Sortino Ratio", "Value": f"{metrics.sortino_ratio():.2f}"},
        {"Metric": "Max Drawdown", "Value": f"{metrics.maximum_dd() * 100:.2f}%"},
        {"Metric": "Total Return", "Value": f"{metrics.total_return() * 100:.2f}%"},
        {"Metric": "CAGR", "Value": f"{metrics.CAGR() * 100:.2f}%"},
        {
            "Metric": "Annualized Volatility",
            "Value": f"{metrics.annualized_volatility() * 100:.2f}%",
        },
        {"Metric": "Profit Factor", "Value": f"{metrics.profit_factor():.2f}"},
        {"Metric": "Win Rate", "Value": f"{metrics.win_rate() * 100:.2f}%"},
        {"Metric": "Expectancy", "Value": f"{metrics.expectancy() * 100:.2f}%"},
        {"Metric": "Max Stagnation Days", "Value": f"{metrics.max_stagnation_days()}"},
        {
            "Metric": "Return/Drawdown Ratio",
            "Value": f"{metrics.return_drawdown_ratio():.2f}",
        },
    ]

    grid = dag.AgGrid(
        rowData=metrics_data,
        columnDefs=[
            {"field": "Metric", "width": 250},
            {"field": "Value", "width": 150},
        ],
        defaultColDef={"resizable": True, "sortable": False},
        dashGridOptions={"domLayout": "autoHeight"},
        style={"height": None},
    )
    return html.Div(
        children=[
            html.H3(
                "Asset Metrics", style={"color": "#00FFCC", "marginBottom": "10px"}
            ),
            grid,
        ]
    )


# Run seasonality analysis on button click
@callback(
    Output("asset-seasonality-output", "children"),
    Input("submit-stats-btn", "n_clicks"),
    State("stats-function-dropdown", "value"),
    State("store_data_H1", "data"),
)
def run_seasonality_analysis(n_clicks, selected_funcs, h1_data):
    if n_clicks == 0 or not selected_funcs or h1_data is None:
        return html.Div()

    # Load and prepare hourly data
    df = pd.DataFrame(h1_data)
    df["DateTime"] = pd.to_datetime(df["DateTime"])
    df = df.set_index("DateTime")

    # Ensure we have enough data
    if df.empty:
        return html.Div("No data available.", style={"color": "#888"})

    season = PlotlySeason(df)

    tabs = []
    reverse_label_map = {v: k for k, v in STATS_FUNCTIONS.items()}

    for func_name in selected_funcs:
        try:
            fig, significant_df = getattr(season, func_name)(show_chart=False)

            # Build AgGrid from significant dataframe
            sig_columns = [
                {"field": col, "sortable": True, "filter": True}
                for col in significant_df.columns
            ]
            sig_rows = significant_df.to_dict("records")

            label = reverse_label_map.get(func_name, func_name)

            tab_content = html.Div(
                children=[
                    dcc.Graph(figure=fig),
                    html.H4(
                        label,
                        style={
                            "color": "#00FFCC",
                            "marginTop": "15px",
                            "marginBottom": "10px",
                        },
                    ),
                    dag.AgGrid(
                        rowData=sig_rows,
                        columnDefs=sig_columns,
                        defaultColDef={
                            "resizable": True,
                            "sortable": True,
                            "filter": True,
                        },
                        dashGridOptions={"domLayout": "autoHeight"},
                        style={"height": None},
                    ),
                ],
                style={"padding": "15px"},
            )

            tabs.append(
                dcc.Tab(
                    label=label,
                    children=[tab_content],
                    style={
                        "backgroundColor": "#1E293B",
                        "color": "#94A3B8",
                        "border": "1px solid #334155",
                        "padding": "8px 16px",
                    },
                    selected_style={
                        "backgroundColor": "#0F172A",
                        "color": "#00FFCC",
                        "borderTop": "2px solid #00FFCC",
                        "fontWeight": "bold",
                    },
                )
            )

        except Exception as e:
            tabs.append(
                dcc.Tab(
                    label=reverse_label_map.get(func_name, func_name),
                    children=[
                        html.Div(
                            f"Error running {func_name}: {str(e)}",
                            style={"color": "#FF6B6B", "padding": "20px"},
                        )
                    ],
                    style={
                        "backgroundColor": "#1E293B",
                        "color": "#94A3B8",
                        "border": "1px solid #334155",
                    },
                    selected_style={
                        "backgroundColor": "#0F172A",
                        "color": "#FF6B6B",
                        "borderTop": "2px solid #FF6B6B",
                    },
                )
            )

    if not tabs:
        return html.Div(
            "No seasonality patterns selected.",
            style={"color": "#888", "marginTop": "10px"},
        )

    return dcc.Tabs(
        children=tabs,
        style={"marginTop": "10px"},
    )


if __name__ == "__main__":
    app.run(debug=True)
