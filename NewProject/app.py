from GettingData import getting_data_mt5
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import plotly.express as px
import seaborn as sns
import plotly.graph_objects as go

from dash import Dash, html, dcc, callback, Output, Input, State
import dash_ag_grid as dag


df = getting_data_mt5("XAUUSD", mt5.TIMEFRAME_D1)
df["Year"] = df.index.year
app = Dash()

app.layout = html.Div(
    children=[
        html.H1(
            "Seasonality Pattern",
            style={"color": "rgb(235, 64, 52)", "textAlign": "center"},
        ),
        dcc.Dropdown(["Open", "High", "Low", "Close"], value="Close", id="Price_value"),
        dcc.Graph(id="figure-output"),
        dcc.Slider(
            min=df["Year"].min(),
            max=df["Year"].max(),
            step=None,
            value=df["Year"].min(),
            marks={str(year): str(year) for year in df["Year"].unique()},
            id="years-slider",
        ),
        html.Br(),
        html.Div(
            children=[
                html.H2("Metrics", style={"textAlign": "center"}),
                dropdown_state := dcc.Dropdown(
                    ["Mean Return", "Median Return", "Max Return", "Min Return"],
                    multi=True,
                    style={"width": "50%"},
                ),
                button_input := html.Button(
                    children="See Results", n_clicks=0, id="button-id"
                ),
                html.Div(id="output-id"),
                html.Div(id="output-id1"),
                html.Div(id="output-id2"),
                html.Div(id="output-id3"),
            ]
        ),
    ]
)


@callback(
    Output("output-id", "children"),
    Output("output-id1", "children"),
    Output("output-id2", "children"),
    Output("output-id3", "children"),
    Input(button_input, "n_clicks"),
    State(dropdown_state, "value"),
)
def getting_metrics(n_clicks, value_metrics):
    mean_return = df["Close"].pct_change().mean()
    median_return = df["Close"].pct_change().median()
    max_return = df["Close"].pct_change().max()
    min_return = df["Close"].pct_change().min()

    out1 = (
        f"Mean Return is {mean_return * 100:.2f}%"
        if value_metrics and "Mean Return" in value_metrics
        else ""
    )
    out2 = (
        f"Median Return is {median_return * 100:.2f}%"
        if value_metrics and "Median Return" in value_metrics
        else ""
    )
    out3 = (
        f"Max Return is {max_return * 100:.2f}%"
        if value_metrics and "Max Return" in value_metrics
        else ""
    )
    out4 = (
        f"Min Return is {min_return * 100:.2f}%"
        if value_metrics and "Min Return" in value_metrics
        else ""
    )

    return out1, out2, out3, out4


@callback(
    Output("figure-output", "figure"),
    Input("Price_value", "value"),
    Input("years-slider", "value"),
)
def drawing_graph(price, selected_year):
    select_df = df[df["Year"] == selected_year]
    fig = px.line(select_df, x=select_df.index, y=price).update_layout(
        xaxis_title="DateTime"
    )
    if price == "Open":
        fig.update_traces(line_color="rgb(252, 3, 3)")
    elif price == "High":
        fig.update_traces(line_color="rgb(3, 161, 252)")
    elif price == "Low":
        fig.update_traces(line_color="yellow")
    else:
        fig.update_traces(line_color=px.colors.qualitative.Plotly[0])
    return fig


if __name__ == "__main__":
    app.run(debug=True)
