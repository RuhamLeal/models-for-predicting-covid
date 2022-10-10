from datetime import datetime
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima.arima import auto_arima
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import re

url = 'https://github.com/neylsoncrepalde/projeto_eda_covid/blob/master/covid_19_data.csv?raw=true'
df = pd.read_csv(url, parse_dates=['ObservationDate', 'Last Update'])

def fix_columns(column_name):
	return re.sub(r"[/| ]", "", column_name).lower()

df.columns = [ fix_columns(column) for column in df.columns ]

brazil = df.loc[
	(df.countryregion == 'Brazil') &
	(df.confirmed > 0)
]
