# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.
import dash
from dash import Dash
import dash_core_components as dcc
import dash_html_components as html

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


app = Dash(__name__)

n_countries, n_years = 185, 18
df1_raw = pd.read_csv('/home/maxcembalest/mysite/data/International_maternal_mortality_country_timeseries.csv')
df1_raw.loc[df1_raw.Country=='United States','Country'] = 'US'
df1_raw.loc[df1_raw.Country=='United Kingdom','Country'] = 'UK'
df1_raw.loc[df1_raw.Country=='Russian Federation','Country'] = 'Russia'
df1_raw.loc[df1_raw.Country=='South Africa','Country'] = 'S. Africa'
df1_raw.loc[df1_raw.Country=='Iran (Islamic Republic of)','Country'] = 'Iran'
df1_raw['pct change'] = (df1_raw['2017 MMR per 100000'] - df1_raw['2000 MMR per 100000'])/df1_raw['2000 MMR per 100000']

df1 = pd.DataFrame(np.zeros(shape=(n_countries*n_years,)))
df1['Year'] = np.array([[x[:4] for x in df1_raw.columns[2:20].values]*n_countries]).flatten()

df1['Country'] = np.array([[x]*n_years for x in df1_raw['Country']]).flatten()
df1['Country ID'] = np.array([[i]*n_years for i in range(n_countries)]).flatten()

df1['mortality'] = np.array([[df1_raw.values[row_num, col_num+2] for col_num in range(n_years)] for row_num in range(n_countries)]).flatten()
df1['logmortality'] = np.log(df1['mortality'])
df1['Relative change in mortality'] = np.array([[df1_raw.values[row_num, col_num+2]/df1_raw.values[row_num, 2] for col_num in range(n_years)] for row_num in range(n_countries)]).flatten() - 1

get_sorted_id = lambda arr, i : np.where(np.argsort(arr)==i)[0][0]
get_sorted_ids = lambda feature : [get_sorted_id(df1_raw[feature], i) for i in df1['Country ID']]
df1['Sorted by 2000 mortality'] = get_sorted_ids('2000 MMR per 100000')
df1['Sorted by 2017 mortality'] = get_sorted_ids('2017 MMR per 100000')
df1['Sorted by pct change in mortality'] = get_sorted_ids('pct change')
df1 = df1.drop([0, 'Country ID'], axis = 1)




fig1 = go.Figure()

countries_to_highlight = \
['France', 'Canada', 'UK', 'US', 'China', 'Norway', 'India', 'Japan', 'Brazil', 'Sierra Leone', 'Afghanistan', 'Kenya', 'S. Africa', 'Iran', 'Iraq', 'Viet Nam']

fig1.add_trace(go.Scatter3d(
        x=df1['Year'],
        y=df1['Sorted by 2017 mortality'],
        z=df1['mortality'],
        text=df1['Country'],
        mode='markers',
        marker=dict(
            size=5,
            color=df1['logmortality'],
            colorscale='reds'
        ),
        hovertemplate=
        "<b>%{text}</b><br><br>" +
        "Year: %{x}<br>" +
        "MMR: %{z}<br>" +
        "<extra></extra>"
        ))

fig1.update_layout(
    title='Countries by 2017 mortality, MMR:= maternal mortality rate per 100,000',
    scene_camera = dict(eye=dict(x=17, y=0, z=15)),
    scene = dict(
        aspectmode = "manual",
        aspectratio = dict( x = 10, y = 30, z = 10),
        xaxis = dict(title='Year'),
        yaxis = dict(
            title='Countries sorted by 2017 MMR',
            tickvals = [get_sorted_id(df1_raw['2017 MMR per 100000'], df1_raw.loc[df1_raw.Country==c].index[0]) for c in countries_to_highlight],
            ticktext=countries_to_highlight
        ),
        zaxis = dict(
            title='MMR',
            tickvals=[0, 500, 1000, 1500, 2000, 2500],
            ticktext=['0', '500', '1000', '1500', '2000', '2500']
        )
    )
)
fig1.update(layout_coloraxis_showscale=False)



fig2 = go.Figure()

countries_to_highlight = \
['France', 'Canada', 'UK', 'US', 'China', 'India', 'Japan', 'Brazil', 'Afghanistan', 'S. Africa', 'Viet Nam', 'Belarus', 'Iran', 'Iraq']


fig2.add_trace(go.Scatter3d(
        x=df1['Year'],
        y=df1['Sorted by pct change in mortality'],
        z=df1['Relative change in mortality'],
        text=df1['Country'],
        mode='markers',
        marker=dict(
            size=5,
            color=df1['Relative change in mortality'],
            colorscale='reds'
        ),
        hovertemplate=
        "<b>%{text}</b><br><br>" +
        "Year: %{x}<br>" +
        "MMR change since 2000: %{z:.2%}<br>" +
        "<extra></extra>"
        ))

fig2.update_layout(
    title='Countries by change in mortality, MMR change:= % change in maternal mortality rate from 2000',
    scene_camera = dict(eye=dict(x=17, y=0, z=15)),
    scene = dict(
        aspectmode = "manual",
        aspectratio = dict( x = 5, y = 30, z = 10),
        xaxis = dict(title='Year'),
        yaxis = dict(
            title='Countries sorted by MMR change',
            tickvals = [get_sorted_id(df1_raw['pct change'], df1_raw.loc[df1_raw.Country==c].index[0]) for c in countries_to_highlight],
            ticktext=countries_to_highlight
        ),
        zaxis = dict(
            title='MMR change',
            tickvals=[-0.5,0,0.5],
            ticktext=['-50%','0%','+50%']
        )
    )
)
fig2.update(layout_coloraxis_showscale=False)



app.layout = html.Div([
    dcc.Graph(figure=fig1),
    dcc.Graph(figure=fig2)
])