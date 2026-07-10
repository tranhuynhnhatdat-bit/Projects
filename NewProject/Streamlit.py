import streamlit as st
import os
from GettingData import getting_data_mt5
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time


@st.cache_data()
def get_data():
    time.sleep(4)
    return getting_data_mt5("XAUUSD", mt5.TIMEFRAME_H1)


st.write("Fetching MT5 Data....")
data = get_data()
st.dataframe(data)
