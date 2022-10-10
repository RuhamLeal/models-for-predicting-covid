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

""" Graph of confirmed cases in Brazil """

px.line(brazil, 'observationdate', 'confirmed', title='Confirmed Cases in Brazil').show()

"""  Creating a column of new covid cases """

brazil['newcases'] = list(map(
	lambda x: 0 if (x==0) else brazil['confirmed'].iloc[x] - brazil['confirmed'].iloc[x-1],
	np.arange(brazil.shape[0])
))

""" Graph of new cases in Brazil """

px.line(brazil, 'observationdate', 'newcases', title='New Cases in Brazil').show()
