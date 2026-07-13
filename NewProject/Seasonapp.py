from PlotlySeason import PlotlySeason
from GettingData import getting_data_mt5
import MetaTrader5 as mt5
from dash import Dash, html, dcc, Output, Input, State, callback
import dash_ag_grid as dag
import plotly.express as px
import plotly.graph_objects as go
import MetaTrader5 as mt5
import pandas as pd
from Metrics import Asset_Metrics

if not mt5.initialize():
    quit()

# Example C: Get ONLY the symbols currently visible in your MT5 Market Watch window
visible_symbols = mt5.symbols_get()
market_watch_list = [s.name for s in visible_symbols if s.visible]


app = Dash()

app.layout = html.Div(
    children=[
        dcc.Store(id="store_data_H1"),
        dcc.Store(id="store_data_1D"),
        html.H1(
            "Seasonality Pattern",
            style={
                "color": "#00FFCC",  # Sleek neon cyan/teal text accent
                "textAlign": "center",
                "fontFamily": "Inter, Helvetica, Arial, sans-serif",
                "fontWeight": "700",  # Makes the title bold and crisp
                "letterSpacing": "1px",  # Adds a premium, modern feel
                "marginBottom": "30px",  # Pushes the dropdown away cleanly
            },
        ),
        dropdown_input := dcc.Dropdown(options=market_watch_list, value="XAUUSD"),
        html.Br(),
        graph_output := dcc.Graph(),
        range_slider_output := dcc.RangeSlider(
            id="years-slider",
        ),
        html.Div(id="asset-metrics-output", style={"marginTop": "30px"}),
    ],
    style={
        "backgroundColor": "#0F172A",  # Midnight navy/slate slate (smooth dark mode)
        "minHeight": "100vh",  # Spans the entire screen height
        "margin": "-8px",  # Removes default white edge gaps
        "padding": "40px 20px",  # Gives elements plenty of breathing room
        "fontFamily": "Inter, Helvetica, Arial, sans-serif",  # Applies modern font everywhere
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


if __name__ == "__main__":
    app.run(debug=True)
