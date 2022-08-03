from dash import Dash
import dash_core_components as dcc
import dash_html_components as html

import numpy as np
import pandas as pd
import itertools
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib import cm
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as clr

app = Dash(__name__)

data_dir = '/home/maxcembalest/mysite/data/'

# assemble energy production & consumption data

global_df = pd.read_csv(data_dir+'global_energy.csv')
global_energy_features = pd.read_csv(data_dir+'global_energy_features.csv')

global_df.loc[global_df.country=='United States','country'] = 'US'
global_df.loc[global_df.country=='United Kingdom','country'] = 'UK'

# filter by these features
my_global_df = global_df.loc[:,[
    'country',
    'iso_code',
    'population',
    'energy_per_capita',
    'gdp',
    'year',
    'coal_consumption',
    'coal_production',
    'electricity_generation',
    'gas_consumption',
    'gas_production',
    'hydro_consumption',
    'low_carbon_consumption',
    'nuclear_consumption',
    'oil_consumption',
    'oil_production',
    'other_renewable_consumption',
    'solar_consumption',
    'wind_consumption'
]]
feature_new_names = {x : x.replace('_', ' ') for x in my_global_df.columns.values}
my_global_df.rename(columns = feature_new_names, inplace = True)

# filter out non-countries
regions = [
    'World', 'Africa', 'Asia Pacific', 'Europe', 'Middle East', 'North America', 'South & Central America', 'CIS'
]
other_regions = [
    'Other CIS', 'Other Asia & Pacific', 'Europe (other)', 'Other Middle East', 'Other Southern Africa'
]
for r in regions + other_regions:
    my_global_df = my_global_df.loc[my_global_df.country != r]

countries = my_global_df.groupby(by='country').count().index
current_df = my_global_df[my_global_df.year==2019]

def get_index_or_nan(c, l):
    try:
        return np.where(l==c)[0][0]+1
    except IndexError:
        return np.nan

def process_energy_stats(energy):
    prod = energy+' production'
    con = energy+' consumption'
    diff = 'net ' + energy

    if energy=='energy': # energy = oil + gas + coal
        my_global_df.loc[:, 'energy production'] = my_global_df['oil production'] + my_global_df['gas production'] + my_global_df['coal production']
        my_global_df.loc[:, 'energy consumption'] = my_global_df['oil consumption'] + my_global_df['gas consumption'] + my_global_df['coal consumption']
        current_df.loc[:, 'energy production'] = current_df['oil production'] + current_df['gas production'] + current_df['coal production']
        current_df.loc[:, 'energy consumption'] = current_df['oil consumption'] + current_df['gas consumption'] + current_df['coal consumption']

    # sort by production and consumption
    sorted_by_current_prod = current_df.sort_values(by = prod)
    sorted_by_current_con = current_df.sort_values(by = con)
    prod_ranked = sorted_by_current_prod.country.values
    con_ranked = sorted_by_current_con.country.values
    my_global_df.loc[:,f'sorted by {prod}'] = [get_index_or_nan(c, prod_ranked) for c in my_global_df.country]
    my_global_df.loc[:,f'sorted by {con}'] = [get_index_or_nan(c, con_ranked) for c in my_global_df.country]

    # measure differential: production - consumption
    my_global_df.loc[:, f'{diff}'] = my_global_df[prod] - my_global_df[con]
    current_df.loc[:,f'{diff}'] = current_df[prod] - current_df[con]
    sorted_by_current_diff = current_df.sort_values(by = diff )
    diff_ranked = sorted_by_current_diff.country.values
    my_global_df.loc[:,f'sorted by {diff}'] = [get_index_or_nan(c, diff_ranked) for c in my_global_df.country]

process_energy_stats('oil')
process_energy_stats('gas')
process_energy_stats('coal')
process_energy_stats('energy')




# custom color maps

def rgba_arr_to_str(r):
    reslist = np.zeros(4)
    reslist[:3] = np.rint(r[:3]*255)
    reslist[3] = r[3]
    res = f'rgba({int(reslist[0])},{int(reslist[1])},{int(reslist[2])},{reslist[3]})'
    return res

def get_custom_rgbas(df, feature, color = 'plasma', opac_low = 1, opac_high = 1):
    if color=='custom':
        my_cmap = clr.LinearSegmentedColormap.from_list('my_cmap', ['#c92222','#41c922'])
    else:
        my_cmap = plt.get_cmap(color)
    norm = mpl.colors.Normalize(vmin=df[feature].min(), vmax=df[feature].max())
    scalarMap = cm.ScalarMappable(norm=norm, cmap=my_cmap)
    my_rgbas = scalarMap.to_rgba(df[feature])
    return [rgba_arr_to_str(my_rgbas[row_num,:]) for row_num in range(my_rgbas.shape[0])]




selected_countries = ['US', 'Russia', 'China']
plot_df = pd.concat([my_global_df[my_global_df.country==c] for c in selected_countries])
feature = 'net energy'

# Initialize figure with subplots
n_rows, n_cols = 1, 1
net_energy_fig = make_subplots(
    rows=n_rows, cols=n_cols,
    column_widths=[1],
    row_heights=[1],
    specs=np.array([[[{"type": "scatter3d"}]*n_cols]*n_rows]).reshape(n_rows, n_cols).tolist(),
    subplot_titles=['Net Energy'],
)

# add a set of time series scatter plots to a fig
def add_multi_time_series_to_fig_subplot(fig, plot_df, height_feature, rownum, colnum, color='plasma'):

    sorted_feature = 'sorted by ' + height_feature

    # sort by sorted feature
    df = plot_df.sort_values(by=[sorted_feature, 'year'])

    # filter out bad values
    df = df.loc[:, ['country', 'year', height_feature, sorted_feature]].fillna(0)
    df = df.loc[df[height_feature] != 0.0]
    df = df.loc[df[sorted_feature] != 0.0]
    current_df = df[df.year==2019]

    # sort by the sorted indices (this lays them out evenly, e.g. 1, 4, 19 --> 0, 1, 2)
    sorted_by_current_sort = current_df.sort_values(by = sorted_feature)
    sort_ranked = sorted_by_current_sort.country.values
    plot_df.loc[:,'sorted by sort'] = [get_index_or_nan(c, sort_ranked) for c in plot_df.country]
    df.loc[:,'sorted by sort'] = [get_index_or_nan(c, sort_ranked) for c in df.country]

    # create custom colors defined above
    df.loc[:,'color'] = get_custom_rgbas(df, height_feature, color=color)

    # add timeseries trace for each country
    for c in df.groupby(by='country').count().index:
        sub_df = df.loc[df.country==c]
        scatter = go.Scatter3d(
            x=sub_df['year'],
            y=sub_df['sorted by sort'],
            z=sub_df[height_feature],
            text=sub_df['country'],
            mode='markers+lines',
            marker=dict(
                size=4,
                color=sub_df.color
            ),
            line=dict(
                width=7,
                color=sub_df.color
            ),
            hovertemplate=
            "   <b>%{text} %{x}</b><br>" +
            f"   {height_feature}<br>"+
            "   %{z:.2f} TWh<br>" +
            "<extra></extra>"
        )
        fig.add_trace(scatter, row=rownum, col=colnum)

def add_config_to_scene(fig, plot_df, scene_name, feature, minval, maxval, z_axis_label):

    fig.update_layout(**{scene_name : {
        'camera' : dict(eye = dict(x=55, y=-55, z=5)),
        'aspectmode' : 'manual',
        'aspectratio' : dict( x = 25, y = 45, z = 25),
        'xaxis' : dict(title='', range = [1980, 2020], tickvals = [1980,2020], ticktext = ['1980', '2020']),
        'yaxis' : dict(
            title = '',
            tickvals = [plot_df.loc[plot_df.country==c, 'sorted by sort'].iloc[0] for c in selected_countries],
            ticktext = selected_countries
        ),
        'zaxis' : dict(
            title = z_axis_label,
            range = [minval, maxval],
            tickvals = np.linspace(minval, maxval, 3)[1:],
            ticktext = [f'{np.round(x/1000, 0):.0f}k TWh' for x in np.linspace(minval, maxval, 3)[1:]]
        )
    }})
    
add_multi_time_series_to_fig_subplot(net_energy_fig, plot_df, feature, 1, 1, color = 'custom')
add_config_to_scene(net_energy_fig, plot_df, 'scene', 'sort', -10000, 10000, 'Net Energy')
net_energy_fig.update_layout(
    template="plotly_dark",
    height = 600,
    margin=dict(r=80, t=50, b=10, l=80),
    showlegend=False
)

# add grid of multi-timeseries plots

selected_countries = ['US', 'Russia', 'China']
plot_df = pd.concat([my_global_df[my_global_df.country==c] for c in selected_countries])

height_features = ['net ' + x for x in ['energy', 'oil', 'gas', 'coal']] + [x+' '+y for y in ['production', 'consumption'] for x in ['energy', 'oil', 'gas', 'coal'] ]
sorted_features = ['sorted by '+ h for h in height_features]

grid_titles = ['Energy                           =', 'Oil                  +', 'Gas                  +', 'Coal             ',
               '', '', '', '',
               '', '', '', '']

# Initialize figure with subplots
n_rows, n_cols = 3, 4
fig = make_subplots(
    rows=n_rows, cols=n_cols,
    column_widths=[0.34, 0.22, 0.22, 0.22],
    row_heights=[0.33, 0.33, 0.33],
    specs=np.array([[[{"type": "scatter3d"}]*n_cols]*n_rows]).reshape(n_rows, n_cols).tolist(),
    subplot_titles=grid_titles,
)

z_axis_titles = ['Net = P - C', '', '', '', 'P (Production)', '', '', '', 'C (Consumption)', '', '', '']

# add multi time series to each subplots
for i, h in enumerate(height_features):
    color='plasma'
    if i//4==0: color = 'custom'
    add_multi_time_series_to_fig_subplot(fig, plot_df, h, i//4 + 1, i%4 + 1, color=color)

    # format each subplot differently
    z_axis_title = z_axis_titles[i]
    scene_name = f'scene{i+1}'
    if i==0: scene_name = 'scene'
    if i < 4: minval, maxval = -10000, 10000
    else: minval, maxval = 0, 40000
    add_config_to_scene(fig, plot_df, scene_name, feature, minval, maxval, z_axis_title)

fig.update_layout(
    template="plotly_dark",
    height = 600,
    margin=dict(r=80, t=50, b=10, l=80),
    showlegend=False
)
grid_fig = fig

# assemble geoplot data

def get_country_coords(c, debug=False):
    if debug: print(c)
    r = country_lat_lon.loc[country_lat_lon.country == c]
    lat, lon = r.lat.iloc[0], r.lon.iloc[0]
    if debug: print('done with ', c)
    return lat, lon

def get_string_map(direc, energy):
    prefix = {'ex' : {'oil' : ('to', 3), 'gas' : ('to', 3)},'im' : {'oil' : ('from', 5), 'gas' : ('From', 5)}}[direc][energy]
    offset = {'oil' : -41, 'gas' : -5}[energy]
    return lambda s : s[s.find(prefix[0])+prefix[1] : offset]

country_lat_lon = pd.read_csv(data_dir+'country_locations.csv')
trade_df_raw_names = {
    'ex' : {'oil' : 'US_Oil_Exports_by_Destination', 'gas' : 'US_Natural_Gas_Exports_and_Re-Exports_by_Country'},
    'im' : {'oil' : 'US_Oil_Imports_by_Country_of_Origin', 'gas' : 'US_Natural_Gas_Imports_by_Country'}
}
energies = ['oil', 'gas']
direcs = ['ex', 'im']
trade_dfs = {direc : {energy : None for energy in energies} for direc in direcs}
country_years = {direc : {energy : None for energy in energies} for direc in direcs}
units = {'oil': 'Mbbl', 'gas' : 'MMcf'}
values = {direc : {energy : f'{energy} {direc}ports (TWh)' for energy in energies} for direc in direcs}

# load different df for exports & imports of each energy
for direc, energy in itertools.product(direcs, energies):

    df_raw = pd.read_csv(data_dir+trade_df_raw_names[direc][energy]+'.csv').fillna(0)
    countries = [get_string_map(direc, energy)(s) for s in df_raw.columns[1:]]
    df = np.zeros(shape = (len(countries)*len(df_raw), 5)).astype(object)
    feature = f'{energy} {direc}ports ({units[energy]})'
    converted_feature = f'{energy} {direc}ports (TWh)'
    df = pd.DataFrame(df, columns = ['country', 'year', feature, 'lat', 'lon'])
    countryyears = np.array([[c, y] for c in countries for y in df_raw.Year])
    df.loc[:,'country'] = countryyears[:,0]
    df.loc[:,'year'] = countryyears[:,1]
    df.loc[:,feature] = df_raw.values[:,1:].T.flatten()
    df.loc[:,converted_feature] = df[feature] / 3412.14163313
    if energy=='oil' : df.loc[:,converted_feature] *= 6
    gascoords = {c : get_country_coords(c) for c in countries}
    df.loc[:,'lat'] = [gascoords[c][0] for c in df.country]
    df.loc[:,'lon'] = [gascoords[c][1] for c in df.country]
    df = df[df[converted_feature] != 0]
    trade_dfs[direc][energy] = df

# add geoplots

current_year = 2020
years = range(2000, current_year + 1)
us_coords = get_country_coords('United States')
maxval = 1200


# Initialize figure with subplots
n_rows, n_cols = 2, 2
fig = make_subplots(
    rows=n_rows, cols=n_cols,
    column_widths=[0.5, 0.5],
    row_heights=[0.5, 0.5],
    specs=np.array([[[{"type": "scattergeo"}]*n_cols]*n_rows]).reshape(n_rows, n_cols).tolist(),
    subplot_titles = ['Oil Exports', 'Natural Gas Exports', 'Oil Imports', 'Natural Gas Imports'],
)

year_lookups = {direc : {energy : {y : [] for y in years} for energy in energies} for direc in direcs}
values = {direc : {energy : f'{energy} {direc}ports (TWh)' for energy in energies} for direc in direcs}
trace_counter = 0
for year in years:

    # filter data
    dfs = {'im' : {'oil' : None, 'gas' : None}, 'ex' : {'oil' : None, 'gas' : None}}
    for direc in direcs:
        for energy in energies:
            df = trade_dfs[direc][energy][trade_dfs[direc][energy].year==str(year)]
            df = df[df[values[direc][energy]] != 0]
            dfs[direc][energy] = df

    # create different geoplot in each subplot for exports & imports of oil & gas
    for k, trade_type in enumerate(['ex', 'im']):
        for j, energy in enumerate(energies):
            df = dfs[trade_type][energy]
            for n in range(len(df)):

                # select data
                row = df.iloc[n]
                country = row['country']
                lat, lon = row[['lat', 'lon']].astype(float)
                val = row[values[trade_type][energy]]

                # display the opacity weighted so lower values are more visible
                # applying the square root achieves approximately that effect
                opacity = np.round(np.sqrt(val / maxval), 3)

                # plot edges
                scatter_line = go.Scattergeo(
                    lon = [us_coords[1], lon],
                    lat = [us_coords[0], lat],
                    mode = 'lines',
                    text=['US', country + ' : ' + np.round(val, 3).astype(str) + ' TWh'],
                    visible = year==current_year,
                    line=dict(
                        width=9*opacity,
                        color=f'rgba(120,120,120,{opacity})'
                    ),
                    hovertemplate=
                    "%{text}"+
                    "<extra></extra>"
                )

                # plot nodes
                scatter_points = go.Scattergeo(
                    lon = [lon],
                    lat = [lat],
                    mode = 'markers',
                    text=[country + ' : ' + np.round(val, 3).astype(str) + ' TWh'],
                    line=dict(color=f'rgba(120,120,120,{opacity + 0.5*(1 - opacity)})'),
                    marker=dict(size=10),
                    visible = year==current_year,
                    hovertemplate=
                    "%{text}"+
                    "<extra></extra>"
                )

                fig.add_trace(scatter_points, row=k+1, col=j+1)
                fig.add_trace(scatter_line, row=k+1, col=j+1)
                year_lookups[trade_type][energy][year] += [trace_counter, trace_counter+1]
                trace_counter+=2

# Create and add slider
steps = []
for year in years:
    step = dict(
        method="update",
        args=[{"visible": [False] * len(fig.data)},
              {"title": "US Oil & Natural Gas Exports & Imports: " + str(year)}],
        label=year
    )
    for trade_type in ['im', 'ex']:
        for energy in energies:
            for i in year_lookups[trade_type][energy][year]:
                step["args"][0]["visible"][i] = True
    steps.append(step)
sliders = [dict(
    active=len(steps) - 1,
    currentvalue={"visible": False},
    pad={"t": 50},
    steps=steps
)]

fig.update_layout(
    template="plotly_dark",
    height = 650,
    sliders=sliders,
)

fig.update_geos(
    projection_type='natural earth',
    landcolor="LightGreen",
    oceancolor="LightBlue",
    showocean=True
)

fig.update(layout_showlegend=False, layout_title = 'US Oil & Natural Gas Exports & Imports: 2020')
export_import_fig = fig


my_style = {'textAlign': 'center', 'color': 'white'}

app.layout = html.Div([
    html.H3(children='The Global Energy Network', style=my_style),
    dcc.Graph(figure=export_import_fig, id='export_import_fig'),
    html.H3(children='Net Energy = Production - Consumption', style=my_style),
    dcc.Graph(figure=net_energy_fig, id='net_energy_fig'),
    html.H3(children='The Global System of Energy Equations', style=my_style),
    dcc.Graph(figure=grid_fig, id='grid_fig'),
    html.H3(children='______________________________', style=my_style),
    ],
    style={'backgroundColor':'rgb(17, 17, 17)'}
)
