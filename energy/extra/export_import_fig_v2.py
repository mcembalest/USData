current_year = 2020
years = range(2000, current_year + 1)
us_coords = get_country_coords('United States')
maxval = 1200


app = dash.Dash()

my_style = {'textAlign': 'center', 'color': 'white'}

app.layout = html.Div([
    html.H3(children='The Global Energy Network', style=my_style),
    dcc.Graph(id='export_import_fig'),
    dcc.Slider(
        2000,
        2020,
        step=None,
        value=2020,
        marks={str(year): str(year) for year in np.arange(2000, 2021)},
        id='year-slider'
    ),
    html.H3(children='Net Energy = Production - Consumption', style=my_style),
    dcc.Graph(figure=net_energy_fig, id='net_energy_fig'),
    html.H3(children='The Global System of Energy Equations', style=my_style),
    dcc.Graph(figure=grid_fig, id='grid_fig'),
    html.H3(children='______________________________', style=my_style),
    ], 
    style={'backgroundColor':'rgb(17, 17, 17)'}
)

@app.callback(
    Output('export_import_fig', 'figure'),
    Input('year-slider', 'value'))
def update_export_import_figure(selected_year):
    
    # filter data
    dfs = {direc : {energy : None for energy in energies} for direc in direcs}
    for direc in direcs:
        for energy in energies:
            df = trade_dfs[direc][energy][trade_dfs[direc][energy].year==str(selected_year)]          
            dfs[direc][energy] = df
    
    # Initialize figure with subplots
    n_rows, n_cols = 2, 2
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        column_widths=[0.5, 0.5],
        row_heights=[0.5, 0.5],
        specs=np.array([[[{"type": "scattergeo"}]*n_cols]*n_rows]).reshape(n_rows, n_cols).tolist(),
        subplot_titles = ['Oil Exports', 'Natural Gas Exports', 'Oil Imports', 'Natural Gas Imports'],
    )
    
    # add geoplots to fig for exports & imports of oil & gas
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
                    hovertemplate=
                    "%{text}"+
                    "<extra></extra>"
                )

                fig.add_trace(scatter_points, row=k+1, col=j+1)
                fig.add_trace(scatter_line, row=k+1, col=j+1)
           
    fig.update_layout(
        template="plotly_dark",
        title = 'US Oil & Natural Gas Exports & Imports',
        height = 600,
        showlegend=False
    )
    fig.update_geos(
        projection_type='natural earth',
        landcolor="LightGreen",
        oceancolor="LightBlue",
        showocean=True
    )
    
    return fig

app.run_server(debug=True, use_reloader=False)