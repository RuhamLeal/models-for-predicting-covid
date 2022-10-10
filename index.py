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

""" Graph of deaths by covid in Brazil #GDCB """

fig_gdcb = go.Figure().add_trace(
	go.Scatter(
		x=brazil.observationdate, y=brazil.deaths, name='Deaths', 
		mode='lines+markers', line={'color': 'red'}
	)
)

fig_gdcb.update_layout(title='Deaths by COVID-19 in Brazil')

fig_gdcb.show()

""" Median growth rate of covid in Brazil """

def growth_rate(data, variable, initial_data=None, last_data=None):
	if initial_data == None:
		initial_data = data.observationdate.loc[data[variable] > 0].min()
	else:
		initial_data = pd.to_datetime(initial_data)
	if last_data == None:
		last_data = data.observationdate.iloc[-1]
	else:
		last_data = pd.to_datetime(last_data)
	
	past = data.loc[data.observationdate == initial_data, variable].values[0]
	present = data.loc[data.observationdate == last_data, variable].values[0]
	points_time = (last_data - initial_data).days
	rate = (present/past)**(1/points_time) - 1

	return rate*100

growth_rate(brazil, 'confirmed')

""" Daily growth rate of covid in Brazil """

def daily_growth_rate(data, variable, initial_data=None):
	if initial_data == None:
		initial_data = data.observationdate.loc[data[variable] > 0].min()
	else:
		initial_data = pd.to_datetime(initial_data)

	last_data = data.observationdate.max()
	points_time = (last_data - initial_data).days

	rates = list(map(
		lambda x: (data[variable].iloc[x] - data[variable].iloc[x-1]) / data[variable].iloc[x-1],
		range(1, points_time + 1)
	))

	return np.array(rates)*100


daily_rate = daily_growth_rate(brazil, 'confirmed')
first_day = brazil.observationdate.loc[brazil.confirmed > 0].min()

px.line(
	x=pd.date_range(first_day, brazil.observationdate.max())[1:],
	y=daily_rate, title='Growth rate of confirmed cases in brazil'
).show()

