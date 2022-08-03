current_year = 2020
years = range(2000, current_year + 1)
us_coords = get_country_coords('United States')

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
# maxval = np.array([trade_dfs[direc][energy][values[direc][energy]].values.max() for direc in direcs for energy in energies]).max()
maxval = 1200
trace_counter = 0
for year in years:
    
    # filter data
    dfs = {'im' : {'oil' : df_oil_im, 'gas' : df_gas_im}, 'ex' : {'oil' : df_oil_ex, 'gas' : df_gas_ex}}
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












reload(logging)
logging.getLogger('werkzeug')
logging.Logger('logger').setLevel(logging.ERROR)

app = dash.Dash()

app.layout = html.Div([
    html.H3(
        children='The Global Energy Network',
        style={
            'textAlign': 'center',
            'color': 'white'
        }
    ),
    dcc.Graph(figure=export_import_fig, id='export_import_fig'),
    html.H3(
        children='The Global System of Energy Equations',
        style={
            'textAlign': 'center',
            'color': 'white'
        }
    ),
    dcc.Graph(figure=grid_fig, id='grid_fig'),
    html.H3(
        children='______________________________',
        style={
            'textAlign': 'center',
            'color': 'white'
        }
    ),
    ], 
    style={'backgroundColor':'rgb(17, 17, 17)'}
)

app.run_server(debug=True, use_reloader=False)