# json library for getting the dictionaries of boroughs and zones
import json 
# logging library to debug the program, to log the required statements to console or specified file
import logging
import os

# numpy library is for arrays and matrices, and high-level mathematical functions 
import numpy as np
# vaex ibrary is for lazy Out-of-Core DataFrames, to visualize and explore big tabular datasets -> our data is of shape (140 million rows * 25 columns)
import vaex

# dash library is for building the web application
import dash
from dash import  dcc, html, dash_table
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

# since we are dealing with large datasets, we cache the required rows for the first run, and use them for later runs
# flask_caching library is used for caching
from flask_caching import Cache

# plotly library is for creating interactive plots of various types
import plotly.express as px
import plotly.graph_objs as go


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('vaex-dash')

external_stylesheets = ['https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server  # used by gunicorn in production mode


cache = Cache(app.server, config={
    'CACHE_TYPE': 'filesystem',
    'CACHE_DIR': 'cache-directory'
})

# set negative to disable (useful for testing/benchmarking)
CACHE_TIMEOUT = int(os.environ.get('DASH_CACHE_TIMEOUT', '60'))


# Get auxiliary information about zones, boroughs, and their relations
with open('./aux_data/taxi_zones-tiny.json') as f:
    geo_json = json.load(f)
with open('./aux_data/zone.json') as f:
    zmapper = json.load(f)
with open('./aux_data/borough.json', 'rb') as f:
    bmapper = json.load(f)
with open('./aux_data/zone_to_borough.json', 'rb') as f:
    zbmapper = json.load(f)

# create dictionaries to map indices to borough and zone names and zone-index to borough-index
zone_index_to_name = {int(index): name for index, name in zmapper.items()}
zone_name_to_index = {name: int(index) for index, name in zmapper.items()}
borough_index_to_name = {int(index): name for index, name in bmapper.items()}
borough_name_to_index = {name: int(index) for index, name in bmapper.items()}
zone_index_to_borough_index = {int(index): borough_name_to_index[zbmapper[name]] for index, name in zmapper.items()}


# open and load the main data
taxi_path = 's3://vaex/taxi/yellow_taxi_2012_zones.hdf5?anon=true'
df_original = vaex.open(taxi_path)

# make sure the data is cached locally, using flask_caching library
used_columns = ['trip_distance',
                'pickup_longitude',
                'pickup_latitude',
                'dropoff_longitude',
                'dropoff_latitude',
                'total_amount',
                'trip_duration_min',
                # 'trip_speed_mph',
                'pickup_hour',
                'pickup_day',
                'dropoff_borough',
                'dropoff_zone',
                'pickup_borough',
                'pickup_zone',
                'tip_amount',
                'fare_amount',
                'vendor_id'
                ]

for col in used_columns:
    print(f'Making sure column "{col}" is cached...')
    df_original.nop(col, progress=True)

# treat these columns as categorical - improves groupby performance.
df_original.categorize(df_original.pickup_day, labels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], inplace=True)
df_original.categorize(df_original.pickup_zone, inplace=True)
df_original.categorize(df_original.dropoff_zone, inplace=True)

# the starting point, initial values for limits of heatmap coordinates, total_amount and duration of trip
heatmap_coords_limits = [[-74.03647198, -73.77135504], [40.6371054, 40.80003402]]
trip_amount_limits = [0, 50]
trip_duration_limits = [0, 50]
num_bins = 25
n_largest = 5
resolution_initial = 75

default_zone = 89  # JFK
default_trip_begin = -73.79413852703125, 40.65619859765626  # JFK
default_trip_end = -73.99194061898439, 40.75039170609375  # Manhatten


# This has to do with layout/styling
fig_layout_defaults = dict(
    plot_bgcolor="#F9F9F9",
    paper_bgcolor="#F9F9F9",
)



overview_md = f'''
### Dataset size {len(df_original):,} rows
The following filters are applied to all of the data
'''
data_summary_filtered_md_template = 'Selected {:,} trips'
data_summary_filtered_md = data_summary_filtered_md_template.format(len(df_original))


def data_bars(df, column):
    ''' function to give styling for histogram data for showing outflow from a selected zone '''
    n_num_bins = 100
    bounds = [i * (1.0 / n_num_bins) for i in range(n_num_bins + 1)]
    ranges = [
        ((df[column].max() - df[column].min()) * i) + df[column].min()
        for i in bounds
    ]
    styles = []
    for i in range(1, len(bounds)):
        min_bound = ranges[i - 1]
        max_bound = ranges[i]
        max_bound_percentage = bounds[i] * 100
        styles.append({
            'if': {
                'filter_query': (
                    '{{{column}}} >= {min_bound}' +
                    (' && {{{column}}} < {max_bound}' if (i < len(bounds) - 1) else '')
                ).format(column=column, min_bound=min_bound, max_bound=max_bound),
                'column_id': column
            },
            'background': (
                """
                    linear-gradient(90deg,
                    #9efa96 0%,
                    #9efa96 {max_bound_percentage}%,
                    white {max_bound_percentage}%,
                    white 100%)
                """.format(max_bound_percentage=max_bound_percentage)
            ),
            'paddingBottom': 2,
            'paddingTop': 2
        })

    return styles




#######################################
# plotly functions to create plots
#######################################
def create_empty_figure():
    ''' creates an empty plotly graph to display while data is being loaded '''
    layout = go.Layout(plot_bgcolor='white', width=10, height=10,
                       xaxis=go.layout.XAxis(visible=False),
                       yaxis=go.layout.YAxis(visible=False))
    return go.Figure(layout=layout)


def create_line_figure():
    ''' function to create line chart of number of trips vs day of the week '''
    df=df_original.copy()
    # gr = df.groupby([df.pickup_borough, df.pickup_day], agg={'count_trips': vaex.agg.count('pickup_datetime')})
    gr = df.groupby([df.pickup_borough, df.pickup_day], agg={'count_trips': vaex.agg.count('pickup_datetime')})

    gr2 = gr.to_pandas_df()
    gr2 = gr2.rename(columns={0: 'pickup_borough', 1: 'pickup_day', 2: 'count_trips'})
    gr2.pickup_borough = gr2.pickup_borough.apply(lambda x: borough_index_to_name[x])
    # print(gr2)
    # pickup_day_index_to_name = {0:'Mon', 1:'Tue', 2:'Wed', 3:'Thu', 4:'Fri', 5:'Sat', 6:'Sun'}
    # gr2.pickup_day = gr2.pickup_day.apply(lambda x: pickup_day_index_to_name[x])
    # print(gr2)

    return px.line(gr2, x='pickup_day', y='count_trips', color='pickup_borough')

def create_bar2_figure():
    ''' function to create a bar plot for payment types with total transactions for each day of the week '''
    df=df_original.copy()

    gr = df.groupby(['payment_type', 'pickup_day'], agg={'total_amount': vaex.agg.sum('total_amount')})
    gr2 = gr.to_pandas_df()
    gr2 = gr2.rename(columns={0: 'payment_type', 1: 'pickup_day', 2: 'total_amount'})

    return px.bar(gr2, x='pickup_day', y='total_amount', color='payment_type', barmode='group')

def create_histogram_figure(x, counts, title=None, xlabel=None, ylabel=None):
    ''' function to create histogram figure for given data, used for number of rides vs total amount or trip duration '''
    
    color = '#9e48e0' # settings
    traces = [] # list of traces

    line = go.scatter.Line(color=color, width=2)
    hist = go.Scatter(x=x, y=counts, mode='lines', line_shape='hv', line=line, name=title, fill='tozerox')
    traces.append(hist)

    title = go.layout.Title(text=title, x=0.5, y=1, font={'color': 'black'})
    margin = go.layout.Margin(l=0, r=0, b=0, t=30)
    legend = go.layout.Legend(orientation='h', bgcolor='#dde0c1', x=0.5, y=1, itemclick=False, itemdoubleclick=False)
    layout = go.Layout(height=230, margin=margin, legend=legend, title=title, xaxis=go.layout.XAxis(title=xlabel), 
                       yaxis=go.layout.YAxis(title=ylabel), **fig_layout_defaults)
    peak = np.round(x[np.argmax(counts)], 2)

    return go.Figure(data=traces, layout=layout), peak


def create_heatmap_figure(data_array, heatmap_limits, trip_start, trip_end):
    ''' function to create a heatmap figure for the nyc data '''
    logger.info("Figure: update heatmap heatmap_limits=%r", heatmap_limits)

    # Set up the layout of the figure
    legend = go.layout.Legend(orientation='h', x=0.0, y=-0.05, font={'color': 'azure'}, bgcolor='royalblue', itemclick=False, itemdoubleclick=False)
    margin = go.layout.Margin(l=0, r=0, b=0, t=30)
    # if we don't explicitly set the width, we get a lot of autoresize events
    layout = go.Layout(height=600, title=None, margin=margin, legend=legend, xaxis=go.layout.XAxis(title='Longitude', range=heatmap_limits[0]), 
                       yaxis=go.layout.YAxis(title='Latitude', range=heatmap_limits[1]), **fig_layout_defaults)
    fig = go.Figure(layout=layout)

    fig = px.imshow(np.log1p(data_array.T), origin='lower')
    fig.layout = layout

    counts = data_array.data
    (xmin, xmax), (ymin, ymax) = heatmap_limits
    dx = (xmax - xmin) / counts.shape[0]
    dy = (ymax - ymin) / counts.shape[1]

    fig.add_trace(go.Heatmap(z=np.log10(counts.T+1), colorscale='plasma', 
                             zmin=None, zmax=None,
                             x0=xmin, dx=(dx),
                             y0=ymin, dy=(dy),
                             showscale=False, hoverinfo=['x', 'y', 'z']))

    # add markers for the points clicked
    def add_point(x, y, **kwargs):
        fig.add_trace(go.Scatter(x=[x], y=[y], marker_color='azure', marker_size=8, mode='markers', showlegend=True, **kwargs))

    if trip_start:
        add_point(trip_start[0], trip_start[1], name='Trip start', marker_symbol='circle')
    if trip_end:
        add_point(trip_end[0], trip_end[1], name='Trip end', marker_symbol='x')

    return fig


def create_geomap_figure(pickup_counts, zone, zoom=10, center={"lat": 40.7, "lon": -73.99}):
    ''' creates a geomap for nyc coloring using the pickup counts '''
    geomap_data = {
        'count': pickup_counts,
        'log_count': np.log10(pickup_counts),
        'zone_name': list(zmapper.values())
    }

    fig = px.choropleth_mapbox(geomap_data,
                               geojson=geo_json,
                               color="log_count",
                               locations="zone_name",
                               featureidkey="properties.zone",
                               mapbox_style="carto-positron",
                               hover_data=['count'],
                               zoom=zoom,
                               center=center,
                               opacity=0.5,
                               )
    # Custom tool-tip
    hovertemplate = '<br>Zone: %{location}' \
                    '<br>Number of trips: %{customdata:.3s}'
    fig.data[0]['hovertemplate'] = hovertemplate

    # draw the selected zone
    geo_json_selected = geo_json.copy()
    geo_json_selected['features'] = [
        feature for feature in geo_json_selected['features'] if feature['properties']['zone'] == zone_index_to_name[zone]
    ]

    geomap_data_selected = {
        'zone_name': [
            geo_json_selected['features'][0]['properties']['zone'],
        ],
        'default_value': ['start'],
        'log_count': [geomap_data['log_count'][zone]],
        'count': [geomap_data['count'][zone]],
    }

    fig_temp = px.choropleth_mapbox(geomap_data_selected,
                                    geojson=geo_json_selected,
                                    color='default_value',
                                    locations="zone_name",
                                    featureidkey="properties.zone",
                                    mapbox_style="carto-positron",
                                    hover_data=['count'],
                                    zoom=9,
                                    center={"lat": 40.7, "lon": -73.99},
                                    opacity=1.,
                                    )
    fig.add_trace(fig_temp.data[0])
    # Custom tool-tip
    hovertemplate = '<br>Zone: %{location}' \
                    '<br>Number of trips: %{customdata:.3s}' \
                    '<extra></extra>'
    fig.data[1]['hovertemplate'] = hovertemplate

    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0}, coloraxis_showscale=False, showlegend=False)
    return fig


def create_sankey_figure(df_outflow_top, df_outflow_rest, df_outflow_borough, pickup_zone):
    ''' creates the flow diagram from flow data for zone to borough to zones (top 5) '''
    pickup_borough = zone_index_to_borough_index[pickup_zone]

    df_outflow_top['dropoff_borough'] = df_outflow_top.dropoff_zone.map(zone_index_to_borough_index)

    label_offset_zones = len(list(bmapper.values()))
    labels = list(bmapper.values()) + list(zmapper.values()) + list([f'Other: {k}' for k in bmapper.values()])

    # overwrite the borough label with one that includes the zone name
    start_name = f'{borough_index_to_name[pickup_borough]} - {zone_index_to_name[pickup_zone]}'
    labels[pickup_borough] = start_name

    source = df_outflow_borough.pickup_borough.astype('int').tolist() + df_outflow_top.dropoff_borough.astype('int').tolist() + df_outflow_rest.dropoff_borough.astype('int').tolist()

    zone_indices = (df_outflow_top.dropoff_zone.astype('int') + label_offset_zones).tolist()
    zone_indices_other = (df_outflow_rest.dropoff_zone.astype('int') + label_offset_zones).tolist()
    target = df_outflow_borough.dropoff_borough.astype('int').tolist() + zone_indices + zone_indices_other
    link_labels = [labels[t] for t in target]

    counts = np.array(df_outflow_borough['count_trips'].tolist() +
                      df_outflow_top['count_trips'].tolist() +
                      df_outflow_rest['count_trips'].tolist())

    line_sankey = go.sankey.node.Line(color='black', width=0.5)
    node_sankey = go.sankey.Node(pad=15, thickness=20, line=line_sankey, label=labels, color='blue')
    link_hovertemplate = 'Origin: %{source.label}<br>Destination %{target.label}'
    link_sankey = go.sankey.Link(source=source, target=target, value=counts, label=link_labels, hovertemplate=link_hovertemplate)
    fig_sankey = go.Figure(data=[go.Sankey(node=node_sankey, link=link_sankey)])

    title_text = f"Outflow of taxis from {zone_index_to_name[pickup_zone]} to other Boroughs, and top {n_largest} zones"
    fig_sankey.update_layout(title_text=title_text, font_size=10, **fig_layout_defaults)
    return fig_sankey

def create_table_data(df_outflow_top):
    df = df_outflow_top.copy()
    last_borough = None
    df['zone'] = df.dropoff_zone.astype('int').map(zone_index_to_name)
    df['borough'] = df.dropoff_zone.map(zone_index_to_borough_index).map(borough_index_to_name)
    records = []
    for i, record in df[['borough', 'zone', 'count_trips']].iterrows():
        if record['borough'] != last_borough:
            last_borough = record['borough']
        else:
            record['borough'] = ''  # don't repeat borough
        records.append(record)

    return records, data_bars(df, 'count_trips')




# ######################################
# computational functions to alter dataframes copies
# ######################################

def create_selection(days, hours, vendors):
    ''' get part of the whole data based on conditions of days, hours, and vendors '''
    df = df_original.copy()
    selection = None
    if hours:
        hour_min, hour_max = hours
        if hour_min > 0:
            df.select((hour_min <= df.pickup_hour), mode='and')
            selection = True
        if hour_max < 23:
            df.select((df.pickup_hour <= hour_max), mode='and')
            selection = True
    if (len(days) > 0) & (len(days) < 7):
        df.select(df.pickup_day.isin(days), mode='and')
        selection = True
    if vendors:
        df.select(df.vendor_id == vendors, mode='and')
        selection = True
    return df, selection


@cache.memoize(timeout=CACHE_TIMEOUT)
def get_heatmap_data(days, hours, vendors, heatmap_limits):
    '''' return the heatmap data array for current selection of days, hours, vendors '''
    logger.info("Compute: heatmap data: days=%r hours=%r vendors=%r limits=%r", days, hours, vendors, heatmap_limits)
    df, selection = create_selection(days, hours, vendors)
    heatmap_data_array = df.count(binby=[df.pickup_longitude, df.pickup_latitude],
                                  selection=selection,
                                  limits=heatmap_limits,
                                  shape=256,
                                  array_type="xarray")
    return heatmap_data_array


@cache.memoize(timeout=CACHE_TIMEOUT)
def get_geomap_data(days, hours, vendors):
    ''' get geomap data with counts for current selection of days, hours, vendors '''
    logger.info("Compute: geomap data: days=%r hours=%r vendors=%r", days, hours, vendors)
    df, selection = create_selection(days, hours, vendors)
    return df.count(binby=df.pickup_zone, selection=selection)


@cache.memoize(timeout=CACHE_TIMEOUT)
def get_trip_details(days, hours, vendors, trip_start, trip_end):
    ''' get the trip details: num_trips, trip_durations for current selection of days, hours, vendors '''
    logger.info("Compute: trip details: days=%r hours=%r vendors=%r trip_start=%r trip_end=%r", days, hours, vendors, trip_start, trip_end)
    df, selection = create_selection(days, hours, vendors)
    # Filter the dataframe
    r = 0.0145 / 20 * 3  # One mile is ~0.0145 deg and 20 blocks per mile.
    pickup_long, pickup_lat = trip_start
    dropoff_long, dropoff_lat = trip_end

    selection_pickup = (df.pickup_longitude - pickup_long)**2 + (df.pickup_latitude - pickup_lat)**2 <= r**2
    selection_dropoff = (df.dropoff_longitude - dropoff_long)**2 + (df.dropoff_latitude - dropoff_lat)**2 <= r**2
    df.select(selection_pickup & selection_dropoff, mode='and')
    selection = True  # after this the selection is always True

    return {
        'counts': df.count(selection=selection),
        'counts_total': df.count(binby=[df.total_amount], limits=[trip_amount_limits], shape=num_bins, selection=selection),
        'counts_duration': df.count(binby=[df.trip_duration_min], limits=[trip_duration_limits], shape=num_bins, selection=selection),
    }


@cache.memoize(timeout=CACHE_TIMEOUT)
def get_flow_data(days, hours, vendors, zone):
    ''' return the sankey flow data, containing flow of taxis from current zone to top 5 destination zones, for current selection of days, hours, vendors '''
    logger.info("Compute: flow data: days=%r hours=%r vendors=%r zone=%r", days, hours, vendors, zone)
    df, selection = create_selection(days, hours, vendors)
    df.select(df.pickup_zone == zone, mode='and')
    selection = True
    df_flow_zone = df.groupby([df.pickup_zone, df.dropoff_zone], agg={'count_trips': vaex.agg.count(selection=selection)})
    # sort descending so we can take the top N
    df_flow_zone = df_flow_zone.sort('count_trips', ascending=False)

    df_flow_zone['pickup_borough'] = df_flow_zone.pickup_zone.map(zone_index_to_borough_index)
    df_flow_zone['dropoff_borough'] = df_flow_zone.dropoff_zone.map(zone_index_to_borough_index)

    pickup_zone = zone
    pickup_borough = zone_index_to_borough_index[pickup_zone]

    # Now to include the total count of all trips for zones that are not the top N
    # only trips leaving from this zone and to a different borough
    df_outflow_zone = df_flow_zone[(df_flow_zone.pickup_zone == pickup_zone)]
    df_outflow_zone = df_outflow_zone[df_outflow_zone.dropoff_borough != pickup_borough]

    df_outflows_top = []
    df_outflows_rest = []

    for dropoff_borough in range(6):
        if dropoff_borough == pickup_borough:
            continue
        # outflow from this zone, to a particular borough
        df_outflow_zone_borough = df_outflow_zone[df_outflow_zone.dropoff_borough == dropoff_borough]
        if len(df_outflow_zone_borough):
            n_max = min(len(df_outflow_zone_borough), n_largest)
            # top N zones of outflow from this zone, to a particular borough
            df_outflows_top.append(df_outflow_zone_borough[:n_max])

            if len(df_outflow_zone_borough) > n_largest:
                count_other = df_outflow_zone_borough[n_largest:]['count_trips'].sum()

                # rest of the outflow from this zone, to a particular borough
                df_outflows_rest.append(vaex.from_scalars(pickup_borough=pickup_borough,
                                        dropoff_borough=dropoff_borough,
                                        dropoff_zone=len(zone_index_to_name) + dropoff_borough,
                                        count_trips=count_other))

    df_outflow_top = vaex.concat(df_outflows_top)
    df_outflow_borough = df_outflow_zone.groupby(['pickup_borough', 'dropoff_borough'],
                                                 agg={'count_trips': vaex.agg.sum('count_trips')}
                                                 )
    if df_outflows_rest:
        df_outflow_rest = vaex.concat(df_outflows_rest)
    else:
        # create an empy dataframe with the same schema to make the rest of the code simpler
        df_outflow_rest = vaex.from_scalars(pickup_borough=-1,
                                            dropoff_borough=-1,
                                            dropoff_zone=-1,
                                            count_trips=-1)[:0]

    # return as dict and lists so it can be serialized by the memoize decorator
    flow_data = dict(
        outflow_top=df_outflow_top.to_dict(array_type='list'),
        outflow_rest=df_outflow_rest.to_dict(array_type='list'),
        outflow_borough=df_outflow_borough.to_dict(array_type='list')
    )
    return flow_data


def get_monthly_averages(df, column='total_amount'):
    ''' for the whole data, we groupby wrt to month and find mean of given column '''
    df=df_original.copy()
    print('#'*30, column)
    avg_column_name = 'avg_'+str(column)
    df['month'] = df['pickup_datetime'].dt.month
    # month_index_to_names = {1:'Jan', 2:'Feb', 3:'Mar', 4:'April', 5:'May', 6:'June', 7:'July', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}
    # df.month = df['month'].apply(lambda x: month_index_to_names[x])
    gr = df.groupby([df.month], agg={avg_column_name: vaex.agg.mean(column)})
    gr2 = gr.to_pandas_df()
    gr2 = gr2.rename(columns={0: 'month', 1: avg_column_name })
    return gr2


# ######################################
# setting up initializations for the project
# ######################################

heatmap_data_initial = get_heatmap_data([], [0, 23], [], heatmap_coords_limits)
geomap_data_initial = get_geomap_data([], [0, 23], [])
trip_detail_data_initial = get_trip_details([], [0, 23], [], default_trip_begin, default_trip_end)
flow_data_initial = get_flow_data([], [0, 23], [], default_zone)

df_outflow_top_initial = vaex.from_dict(flow_data_initial['outflow_top'])
df_outflow_rest_initial = vaex.from_dict(flow_data_initial['outflow_rest'])
df_outflow_borough_initial = vaex.from_dict(flow_data_initial['outflow_borough'])

figure_sankey_initial = create_sankey_figure(df_outflow_top_initial, df_outflow_rest_initial, df_outflow_borough_initial, default_zone)

table_records_intitial, table_style_initial = create_table_data(df_outflow_top_initial)

zone_summary_template_md = '''
**{}**: **{:,}** taxi trips leaving this zone.

_Click on the map to change the zone._
'''

zone_pickup_count = geomap_data_initial[default_zone]
zone_summary_md = zone_summary_template_md.format(zone_index_to_name[default_zone], zone_pickup_count, len(df_original))



# ######################################
# dash part of the code, setting up the app layout to model our plots in div containers using dash.dcc, dbc etc..
# ######################################

# The app layout
app.layout = html.Div(className='app-body', children=[
    # Stores
    dcc.Store(id='map_clicks', data=0),
    dcc.Store(id='zone', data=default_zone),
    dcc.Store(id='trip_start', data=default_trip_begin),
    dcc.Store(id='trip_end', data=default_trip_end),
    dcc.Store(id='heatmap_limits', data=heatmap_coords_limits),
    
    # Control panel
    html.Div([
        dbc.Row([
            dcc.Loading(
                className="loader",
                id="loading",
                type="default",
                children=[
                    html.Div(id='loader-trigger-1', style={"display": "none"}),
                    html.Div(id='loader-trigger-2', style={"display": "none"}),
                    html.Div(id='loader-trigger-3', style={"display": "none"}),
                    html.Div(id='loader-trigger-4', style={"display": "none"}),
                    dcc.Markdown(id='data_summary_filtered', children=data_summary_filtered_md),
                    html.Progress(id="selected_progress", max=f"{len(df_original)}", value=f"{len(df_original)}"),
                ]
            ),
        ]),
        dbc.Row([
            dbc.Col(className="four columns pretty_container", children=[
                html.Label('Select Vendor'),
                dcc.Dropdown(id='vendors',
                            placeholder='Select vendor of taxis',
                            options=[{'label': 'VTS', 'value': 'VTS'},
                                    {'label': 'CMT', 'value': 'CMT'}],
                            value=[]
                            ),
            ], width=4),
            dbc.Col([
                html.Label('Select pick-up hours'),
                dcc.RangeSlider(id='hours',
                                value=[0, 23],
                                min=0, max=23,
                                marks={i: str(i) for i in range(0, 24, 3)}),
            ], width=4),
            dbc.Col([
                html.Label('Select pick-up days'),
                dcc.Dropdown(id='days',
                            placeholder='Select a day of week',
                            options=[{'label': 'Monday', 'value': 0},
                                    {'label': 'Tuesday', 'value': 1},
                                    {'label': 'Wednesday', 'value': 2},
                                    {'label': 'Thursday', 'value': 3},
                                    {'label': 'Friday', 'value': 4},
                                    {'label': 'Saturday', 'value': 5},
                                    {'label': 'Sunday', 'value': 6}],
                            value=[],
                            multi=True),
            ], width=5),
        ]),
    ]),
    # The Visuals
    html.Div(id='tab', children=[
        html.Div(children=[
            dcc.Markdown(children=['Popular destinations in New York']),
            dbc.Row([
                dbc.Col([
                    dcc.Markdown(id='zone_summary', children=zone_summary_md),
                    dcc.Graph(id='geomap_figure',
                              figure=create_geomap_figure(geomap_data_initial, default_zone),
                              config={"modeBarButtonsToRemove": ['lasso2d', 'select2d']})
                ], width=12),
            ]),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='flow_sankey_figure',
                              figure=figure_sankey_initial,
                              config={"modeBarButtonsToRemove": ['lasso2d', 'select2d']})
                ], width=12),
                dbc.Col([
                    dash_table.DataTable(id='table', columns=[
                            {'name': 'Destination Borough', 'id': 'borough'},
                            {'name': 'Destination zone', 'id': 'zone'},
                            {'name': 'Number of trips', 'id': 'count_trips'},
                        ],
                        data=table_records_intitial,
                        style_data_conditional=table_style_initial,
                        style_as_list_view=True,
                    )
                ], width=12),
            ]),
        ]),
    ]),

    html.Div(children=[
        dbc.Row([
            dbc.Col([
                dcc.Markdown(children='_Click on the map to select trip start and destination._'),
                dcc.Graph(id='heatmap_figure',
                            figure=create_heatmap_figure(heatmap_data_initial, heatmap_coords_limits, default_trip_begin, default_trip_end),
                            config={"modeBarButtonsToRemove": ['lasso2d', 'select2d', 'hoverCompareCartesian']})
            ], width=7),
            dbc.Col([
                        dcc.Graph(id='trip_summary_amount_figure'),
                        dcc.Graph(id='trip_summary_duration_figure'),
                        dcc.Markdown(id='trip_summary_md'),
            ], width=5)
        ]),
    ]),
    dbc.Row([
        dbc.Col(className="four columns pretty_container", children=[
            html.Label('Select Property'),
            dcc.Dropdown(id='prop',
                         placeholder='Select property to show monthly average',
                         options=[{'label': 'total amount', 'value': 'total_amount'},
                                  {'label': 'fare amount', 'value': 'fare_amount'},
                                  {'label': 'tip amount', 'value': 'tip_amount'},
                                  {'label': 'trip distance', 'value': 'trip_distance'},
                                  {'label': 'trip duration', 'value': 'trip_duration_min'}
                                  ],
                         value=[]
                        ),
        ], width=4),
        dbc.Col(html.Div(children=[
            dcc.Markdown(children='Line Charts of Fare, Tip, Total Amounts of taxi rides across the months'),
            dcc.Graph(
                id='line_charts_figure'
            ),
        ])
        )]
    ),
    html.Div(children=[
        dbc.Row([
            dbc.Col([
                dcc.Graph(figure=create_line_figure())
            ], width=6),
            dbc.Col([
                dcc.Graph(figure=create_bar2_figure())
            ], width=6)
        ])
    ]),
])


@app.callback(
    Output('line_charts_figure', 'figure'),
    Input('prop', 'value'),
)
def update_line_chart(prop):
    column = 'total_amount'
    column = prop
    monthly_averages = get_monthly_averages(df_original, column)
    month_names = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun", 7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
    fig = px.line(monthly_averages, x='month', y='avg_'+column)
    fig.update_layout(
        title='Monthly Averages of ' + column,
        xaxis_title='Month',
        yaxis_title='Average of ' + column,
        xaxis = {
            "tickvals": list(range(1, 13)), 
            "ticktext": [month_names[i] for i in range(1, 13)]
            }
        )

    return fig

# Heatmap related computation callbacks

# Map figure
@app.callback([Output('heatmap_figure', 'figure'),
               Output('loader-trigger-4', 'children')],
              [Input('days', 'value'),
               Input('hours', 'value'),
               Input('vendors', 'value'),
               Input('heatmap_limits', 'data'),
               Input('trip_start', 'data'),
               Input('trip_end', 'data')],
              prevent_initial_call=True)
def update_heatmap(days, hours, vendors, heatmap_limits, trip_start, trip_end):
    data_array = get_heatmap_data(days, hours, vendors, heatmap_limits)
    return create_heatmap_figure(data_array, heatmap_limits, trip_start, trip_end), "trigger loader"


# HeatMap navigation (zoom/pan)
@app.callback(
    Output('heatmap_limits', 'data'),
    [Input('heatmap_figure', 'relayoutData')],
    [State('heatmap_limits', 'data')],
    prevent_initial_call=True)
def update_heatmap_limits(relayoutData, heatmap_limits):
    print("\n\n\nRelayOut Data ")
    print(relayoutData)
    print("\n\n\n")
    logger.info('Interaction: map/zoom on heatmap detected: relayoutData=%r heatmap_limits=%r', relayoutData, heatmap_limits)
    if relayoutData is None:
        raise dash.exceptions.PreventUpdate
    elif relayoutData is not None and 'xaxis.range[0]' in relayoutData:
        d = relayoutData
        heatmap_limits = [[d['xaxis.range[0]'], d['xaxis.range[1]']], [d['yaxis.range[0]'], d['yaxis.range[1]']]]
    else:
        raise dash.exceptions.PreventUpdate
        if heatmap_limits is None:
            heatmap_limits = heatmap_coords_limits
    return heatmap_limits


# Heatmap interaction (clicking)
@app.callback([Output('map_clicks', 'data'),
               Output('trip_start', 'data'),
               Output('trip_end', 'data')],
              [Input('heatmap_figure', 'clickData')],
              [State('map_clicks', 'data'),
               State('trip_start', 'data'),
               State('trip_end', 'data')],
              prevent_initial_call=True)
def on_click_heatmap(click_data_heatmap, map_clicks, trip_start, trip_end):
    logger.info('Interaction: click on heatmap detected: %r', click_data_heatmap)
    if click_data_heatmap is not None:
        point = click_data_heatmap['points'][0]['x'], click_data_heatmap['points'][0]['y']
        new_location = point[0], point[1]
        # the 1st and 3rd and 5th click change the start point
        if map_clicks % 2 == 0:
            trip_start = new_location
            trip_end = None  # and reset the end point
        else:
            # the 2nd, 4th etc set the end point
            trip_end = new_location
        map_clicks += 1
    logger.info('Interaction: trip_start=%r trip_end=%r', trip_start, trip_end)
    return map_clicks, trip_start, trip_end


# Geographical map

# Geographical map click on geojson area
@app.callback(Output('zone', 'data'),
              [Input('geomap_figure', 'clickData'),
               Input('flow_sankey_figure', 'clickData'),
               Input('table', 'active_cell')],
              [State('zone', 'data'),
               State('table', 'data')],
              prevent_initial_call=True)
def on_click(click_data_geomap, click_data_sunkey, click_data_table, zone, table_data):
    trg = dash.callback_context.triggered   # What triggered the callback
    logger.info('Interaction: click on popular destinations tab detected: %r', trg)

    if trg is not None:
        component = trg[0]['prop_id'].split('.')[0]

        if component == 'geomap_figure':
            zone_name = trg[0]['value']['points'][0]['location']
            zone = zone_name_to_index[zone_name]
        if component == 'flow_sankey_figure':
            zone_name = trg[0]['value']['points'][0]['label']
            if zone_name in zone_name_to_index.keys():
                zone = zone_name_to_index[zone_name]
            else:
                raise dash.exceptions.PreventUpdate()
        if component == 'table':
            if trg[0]['value']['column_id'] == 'zone':
                table_row = trg[0]['value']['row']
                zone_name = table_data[table_row]['zone']
                zone = zone_name_to_index[zone_name]
            else:
                raise dash.exceptions.PreventUpdate()
    return zone


# Geographical map data
@app.callback([Output('geomap_figure', 'figure'),
               Output('data_summary_filtered', 'children'),
               Output('zone_summary', 'children'),
               Output('selected_progress', 'value'),
               Output('loader-trigger-3', 'children')],
              [Input('days', 'value'),
               Input('hours', 'value'),
               Input('vendors', 'value'),
               Input('zone', 'data')],
              [State('geomap_figure', 'figure')],
              prevent_initial_call=True
              )
def update_geomap(days, hours, vendors, zone, current_figure):
    logger.info('Figure: updating geo map for: days=%r hours=%r vendors=%r zone=%r', days, hours, vendors, zone)

    zoom = current_figure['layout']['mapbox']['zoom']
    center = current_figure['layout']['mapbox']['center']

    pickup_counts = get_geomap_data(days, hours, vendors)
    fig = create_geomap_figure(pickup_counts, zone, zoom=zoom, center=center)

    # we piggy back on the calculated pickup_counts to calculate what are the # filtered rows
    # instead of doing another calculation / pass over the data
    count = pickup_counts.sum()
    markdown_text = data_summary_filtered_md_template.format(count)

    zone_pickup_count = pickup_counts[zone]
    zone_summary_md = zone_summary_template_md.format(zone_index_to_name[zone], zone_pickup_count, len(df_original))

    return fig, markdown_text, zone_summary_md, str(count), "trigger loader"


# Flow section
@app.callback(
    [Output('flow_sankey_figure', 'figure'),
     Output('table', 'data'),
     Output('table', 'style_data_conditional'),
     Output('loader-trigger-1', 'children')
     ],
    [Input('days', 'value'),
     Input('hours', 'value'),
     Input('vendors', 'value'),
     Input('zone', 'data'),
     ], prevent_initial_call=True
)
def update_sankey(days, hours, vendors, zone):
    logger.info('Figure: update sankey for days=%r hours=%r vendors=%r zone=%r', days, hours, vendors, zone)
    flow_data = get_flow_data(days, hours, vendors, zone)
    df_outflow_top = vaex.from_dict(flow_data['outflow_top'])
    df_outflow_rest = vaex.from_dict(flow_data['outflow_rest'])
    df_outflow_borough = vaex.from_dict(flow_data['outflow_borough'])

    pickup_zone = zone
    fig_sankey = create_sankey_figure(df_outflow_top, df_outflow_rest, df_outflow_borough, pickup_zone)
    table_records, table_style = create_table_data(df_outflow_top)

    return fig_sankey, table_records, table_style, 'trigger loader'


# Trip plotting

@app.callback([Output('trip_summary_amount_figure', 'figure'),
               Output('trip_summary_duration_figure', 'figure'),
               Output('trip_summary_md', 'children'),
               Output('loader-trigger-2', 'children')],
              [Input('days', 'value'),
               Input('hours', 'value'),
               Input('vendors', 'value'),
               Input('trip_start', 'data'),
               Input('trip_end', 'data')]
              )
def update_trip_details_summary(days, hours, vendors, trip_start, trip_end):
    if trip_start is None or trip_end is None:
        fig_empty = create_empty_figure()
        if trip_start is None:
            text = '''Please select a start location on the map.'''
        else:
            text = '''Please select a destination location on the map.'''
        return fig_empty, fig_empty, text, "trigger loader"

    trip_detail_data = get_trip_details(days, hours, vendors, trip_start, trip_end)
    logger.info('Figure: trip details summary for %r to %r', trip_start, trip_end)

    counts = trip_detail_data['counts']
    counts_total = np.array(trip_detail_data['counts_total'])
    counts_duration = np.array(trip_detail_data['counts_duration'])
    fig_amount, peak_amount = create_histogram_figure(df_original.bin_edges(df_original.total_amount, trip_amount_limits, shape=num_bins),
                                                      counts_total,
                                                      title=None,
                                                      xlabel='Total amount [$]',
                                                      ylabel='Numbe or rides')
    
    fig_duration, peak_duration = create_histogram_figure(df_original.bin_edges(df_original.trip_duration_min, trip_amount_limits, shape=num_bins),
                                                          counts_duration,
                                                          title=None,
                                                          xlabel='Trip duration [min]',
                                                          ylabel='Numbe or rides')

    trip_stats = f'''
                    **Trip statistics:**
                    - Number of rides: {counts}
                    - Most likely trip total cost: ${peak_amount}
                    - Most likely trip duration: {peak_duration} minutes
                    '''

    return fig_amount, fig_duration, trip_stats, "trigger loader"


if __name__ == '__main__':
    app.run_server(port=8080)