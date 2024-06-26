import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pandas as pd
from plotly.subplots import make_subplots
from ML_Preparation.Feature_Engineering import *
import tensorflow as tf
from Forecasting_Evaluation.Evaluator import Evaluator
import plotly.express as px
from Deployment.ML_Deployment import *
from dash import dash_table

# Load data
df = pd.read_parquet("df_preprocessed_no_sample.parquet")   # dataframe with no rolling window and no sampling greater than 1 hour (sample_time = rolling_window = "1min")
df_forecast = pd.read_parquet("df_preprocessed.parquet")    # dataframewith rolling window 3d and sample time 1d
df_raw = pd.read_parquet("df_raw.parquet")  #raw dataframe without any preprocessing


df["date"] = df["date_time"].dt.date
df["hour"] = df["date_time"].dt.hour

sensor_data_liste = ['tmp', 'hum', 'CO2', 'VOC', 'vis', 'IR', 'BLE']
room_numbers = df['room_number'].sort_values().unique()
n_steps = 7

# loading Forecast
fe = FeatureEngineering(df_forecast,
                        label = "CO2", 
                        categorical_features = ["season", "room_number", "dayofweek"],
                        automated_feature_engineering = False)
X_train, X_val, X_test, y_train, y_val, y_test = fe.feature_engineering(steps_to_forecast = n_steps, skip_scale = True)
model = tf.keras.models.load_model(f"CO2_Forecasting_Model.keras")
pred = model.predict(fe.X_test)
fe.df = df_raw.copy()
deployer = Predictor(data = df_raw, feature_engineering_class_object = fe, label = "CO2", is_forecast = True, roll = True, steps_to_forecast = 2)
forecasted_pred = deployer.predict(x = deployer.x, model = model)
forecasted_pred.reset_index(inplace=True)
forecasted_pred = forecasted_pred[forecasted_pred["date_time"] == df_forecast.date_time.max()].sort_values("room_number")

ev = Evaluator()

#plotting forecast

fig_forecast = px.line(y = [fe.y_test[:, 0], pred[:, 0]], 
            labels = {"wide_variable_0": "y_true",
                    "wide_variable_1": "Modell 1 - Vorhersage (Wetterdaten + Jahreszeiten)"})

fig_forecast.update_layout(
        colorway=["#5E0DAC", '#FF4F00', '#375CB1', '#FF7400', '#FFF400', '#FF0056'],
        template='plotly_dark',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        plot_bgcolor='rgba(0, 0, 0, 0)',
        margin={'b': 15},
        hovermode='x',
        autosize=True,
        title={'text':  f"Forecast für CO2(t + {0})  blau = y_true, rot = y_pred", 'font': {'color': 'white'}, 'x': 0.5},           
    )
# Initialize the app
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.config.suppress_callback_exceptions = True

def get_options(sensor_data):
    return [{'label': i, 'value': i} for i in sensor_data]

options = get_options(sensor_data_liste)
room_options = [{'label': 'Keine Filterung', 'value': 'all'}] + get_options(room_numbers)
# Layout for the sidebar
sidebar_layout = html.Div(
    children=[
        html.H2('Navigation'),
        html.Ul(
            children=[
                html.Li(dcc.Link('Übersicht', href='/uebersicht')),
                html.Li(dcc.Link('Monatsübersicht', href='/monatsuebersicht')),
                html.Li(dcc.Link('Tagesübersicht', href='/tagesuebersicht')),
                html.Li(dcc.Link('Forecast', href='/forecast'))
            ]
        )
    ],
    style={'padding': '10px', 'width': '20%', 'display': 'inline-block', 'verticalAlign': 'top'}
)

# Layout for the main content
app.layout = html.Div(
    children=[
        dcc.Location(id='url', refresh=True),
        sidebar_layout,
        html.Div(id='page-content', style={'padding': '10px', 'width': '75%', 'display': 'inline-block', 'verticalAlign': 'top'})
    ]
)

# Callback to update the page content
@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/monatsuebersicht':
        return monatsuebersicht_layout
    elif pathname == '/tagesuebersicht':
        return tagesuebersicht_layout
    elif pathname == '/forecast':
        return forecast_layout
    else:
        return uebersicht_layout

# Layout uebersicht page
uebersicht_layout = html.Div(
    children=[
        html.H1('Übersicht'),
        html.Div(
            children=[
                html.P('Wähle räumliche Filteroptionen:'),
                dcc.Dropdown(id='count-filter-selector', options=room_options,
                             multi=True,
                             style={'backgroundColor': '#1E1E1E', 'width': '80%', 'display': 'inline-block'},
                             className='count-filterselector'
                )
            ],
            style={'display': 'flex', 'alignItems': 'center'}
        ),
        dcc.Graph(id='count-timeseries', config={'displayModeBar': False}, animate=False, style={'backgroundColor': '#2D2D2D'}),
    ]
)
# Layout for Monatsübersicht
monatsuebersicht_layout = html.Div(
    children=[
        html.H1('Monatsübersicht'),
        html.Div(
            children=[
                html.P('Wähle ein oder zwei Sensordaten, die du visualisieren willst:'),
                dcc.Dropdown(id='month-sensordataselector', options=options,
                             multi=True, value=[sensor_data_liste[0]],
                             style={'backgroundColor': '#1E1E1E', 'width': '48%', 'display': 'inline-block'},
                             className='month-sensordataselector'
                ),
                html.Div(id='month-input-warning', style={'display': 'inline-block', 'width': '4%'}),
                html.P('Wähle räumliche Filteroptionen:'),
                dcc.Dropdown(id='month-filter-selector', options=room_options,
                             multi=True, value=['all'],
                             style={'backgroundColor': '#1E1E1E', 'width': '80%', 'display': 'inline-block'},
                             className='month-filterselector'
                )
            ],
            style={'display': 'flex', 'alignItems': 'center'}
        ),
        dcc.Graph(id='month-timeseries', config={'displayModeBar': False}, animate=False, style={'backgroundColor': '#2D2D2D'}),
    ]
)

# Layout for Tagesübersicht
tagesuebersicht_layout = html.Div(
    children=[
        html.H1('Tagesübersicht'),
        html.Div(
            children=[
                html.P('Wähle ein oder zwei Sensordaten, die du visualisieren willst:'),
                dcc.Dropdown(id='sensordataselector', options=options,
                             multi=True, value=[sensor_data_liste[0]],
                             style={'backgroundColor': '#1E1E1E', 'width': '48%', 'display': 'inline-block'},
                             className='sensordataselector'
                ),
                html.Div(id='input-warning', style={'display': 'inline-block', 'width': '4%'}),
                html.P('Wähle räumliche Filteroptionen:'),
                dcc.Dropdown(id='filter-selector', options=room_options,
                             multi=True, value=['all'],
                             style={'backgroundColor': '#1E1E1E', 'width': '48%', 'display': 'inline-block'},
                             className='filterselector'
                )
            ],
            style={'display': 'flex', 'alignItems': 'center'}
        ),
        dcc.Graph(id='timeseries', config={'displayModeBar': False}, animate=False, style={'backgroundColor': '#2D2D2D'}),
        html.Br(),
        html.H1('Tagesübersicht für angegebenen Tag'),
        dcc.Input(id='time-input', type='text', value=str(df["date"].max()), 
                  style={'backgroundColor': '#1E1E1E', 'color':'white'},
                  className='dateinput'),
        dcc.Graph(id='daily-timeseries', config={'displayModeBar': False}, animate=False, style={'backgroundColor': '#2D2D2D'})
    ]
)

#Layout for forecast page
forecast_layout = html.Div(
    children=[
        html.H1('Forecast'),
        dcc.Graph(id='forecastgraph',
                  figure= fig_forecast, 
                  config={'displayModeBar': False}, animate=False,
                  style={'backgroundColor': '#2D2D2D'}),
        html.Div(id='input-warning', style={'display': 'inline-block', 'width': '4%'}),
        html.H1('CO2-Forecast für die nächsten 7 Tage'),
        dash_table.DataTable(
            id='table',
            columns=[{"name": i, "id": i} for i in forecasted_pred.columns],
            data=forecasted_pred.to_dict('records'),
            style_cell={'backgroundColor': '#2D2D2D', 'color': 'white'},
            style_header={'backgroundColor': '#1E1E1E', 'fontWeight': 'bold'},
            style_table={'height': '300px', 'overflowY': 'auto'})
    ]
)
    
@app.callback(
    [Output("input-warning", 'children'), 
     Output("filter-selector", "options"), 
     Output("filter-selector", "value"),
     Output("filter-selector", "disabled"),
     Output("filter-selector", "multi")], 
    [Input("sensordataselector", "value")]
)
def update_filter_options(sensordata):
    """update filteroptions, depending on what is selected"""

    warning = None
    room_disabled = False
    multi_select = True

    # only allow 2 sensor data choices
    if len(sensordata) > 1:
        room_disabled = False  # allow choice   
        multi_select = False  # allow multi select  
        selected_value = 'all'  # choose 'all' as default value
    else:
        selected_value = ['all']  # default value
    return [warning, room_options, selected_value, room_disabled, multi_select]

@app.callback(Output('timeseries', 'figure'),
              [Input('filter-selector', 'value'),
               Input('sensordataselector', 'value')])
def update_timeseries(selected_filters, selected_dropdown_value):
    ''' lineplot for daily aggregation '''
    if len(selected_dropdown_value) > 2:
        selected_dropdown_value = selected_dropdown_value[:2]

    # convert to list if neccessary
    if not isinstance(selected_filters, list):
        selected_filters = [selected_filters]

    # create fig object
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # check if all is in selected filter, so that traces can be combined
    if 'all' in selected_filters:
        for data in selected_dropdown_value:
            data_daily_all = df.groupby('hour', as_index=False).agg({data: "mean"})
            fig.add_trace(
                go.Scatter(x=data_daily_all["hour"],
                           y=data_daily_all[data],
                           mode='lines',
                           opacity=0.7,
                           name=f'{data} (alle Räume)',
                           textposition='bottom center'),
                secondary_y=False if data == selected_dropdown_value[0] else True)

    # add traces for room numbers
    
    for room in selected_filters:
        room_data = df[df['room_number'] == room]
        for data in selected_dropdown_value:
            room_daily = room_data.groupby('hour', as_index=False).agg({data: "mean"})
            # specific name for every trace
            trace_name = f'{data} (alle Räume)' if room == 'all' else f'{data} (Raum {room})'
            fig.add_trace(
                go.Scatter(x=room_daily["hour"],
                        y=room_daily[data],
                        mode='lines',
                        opacity=0.7,
                        name=trace_name,
                        textposition='bottom center'),
                secondary_y=False if data == selected_dropdown_value[0] else True)
            
    # update layout
    fig.update_layout(
        colorway=["#5E0DAC", '#FF4F00', '#375CB1', '#FF7400', '#FFF400', '#FF0056'],
        template='plotly_dark',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        plot_bgcolor='rgba(0, 0, 0, 0)',
        margin={'b': 15},
        hovermode='x',
        autosize=True,
        title={'text': 'Tagesübersicht', 'font': {'color': 'white'}, 'x': 0.5},           
    )
    return fig

@app.callback(
    Output('daily-timeseries', 'figure'),
    [Input('time-input', 'value'),
     Input('filter-selector', 'value'),
     Input('sensordataselector', 'value')]
)
def update_daily_timeseries(selected_date, selected_filters, selected_dropdown_value):
    ''' Draw traces of the feature 'value' for the specified date and filter '''
    if len(selected_dropdown_value) > 2:
        selected_dropdown_value = selected_dropdown_value[:2]

    # Ensure the selected date is valid
    try:
        selected_date = pd.to_datetime(selected_date).date()
        df_filtered = df[df['date'] == selected_date]
    except ValueError:
        return go.Figure()  # Return an empty figure if the date is invalid

    # convert to list, if neccessary
    if not isinstance(selected_filters, list):
        selected_filters = [selected_filters]

    # Create the figure object
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add traces for all rooms if 'all' is in the selected filters
    if 'all' in selected_filters:
        for data in selected_dropdown_value:
            data_daily_all = df_filtered.groupby('hour', as_index=False).agg({data: "mean"})
            fig.add_trace(
                go.Scatter(x=data_daily_all["hour"],
                           y=data_daily_all[data],
                           mode='lines',
                           opacity=0.7,
                           name=f'{data} (alle Räume)',
                           textposition='bottom center'),
                secondary_y=False if data == selected_dropdown_value[0] else True)

    # Add traces for the selected rooms
    for room in selected_filters:
        room_data = df_filtered[df_filtered['room_number'] == room]
        for data in selected_dropdown_value:
            room_daily = room_data.groupby('hour', as_index=False).agg({data: "mean"})
            trace_name = f'{data} (alle Räume)' if room == 'all' else f'{data} (Raum {room})'
            fig.add_trace(
                go.Scatter(x=room_daily["hour"],
                           y=room_daily[data],
                           mode='lines',
                           opacity=0.7,
                           name=trace_name,
                           textposition='bottom center'),
                secondary_y=False if data == selected_dropdown_value[0] else True)

    # Update the layout of the plot
    fig.update_layout(
        colorway=["#5E0DAC", '#FF4F00', '#375CB1', '#FF7400', '#FFF400', '#FF0056'],
        template='plotly_dark',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        plot_bgcolor='rgba(0, 0, 0, 0)',
        margin={'b': 15},
        hovermode='x',
        autosize=True,
        title={'text': f'Tagesübersicht für {selected_date}', 'font': {'color': 'white'}, 'x': 0.5},
    )

    return fig

@app.callback(
    [Output("month-input-warning", 'children'), 
     Output("month-filter-selector", "options"), 
     Output("month-filter-selector", "value"),
     Output("month-filter-selector", "disabled"),
     Output("month-filter-selector", "multi")], 
    [Input("month-sensordataselector", "value")]
)
def update_monthly_filter_options(sensordata):
    """updating filteroptions depending on what is selected"""
    warning = None
    room_disabled = False
    multi_select = True

    
    if len(sensordata) > 1:

        room_disabled = False  
        multi_select = False  
        selected_value = 'all'  
    else:
        selected_value = ['all']  

    return [warning, room_options, selected_value, room_disabled, multi_select]

@app.callback(Output('month-timeseries', 'figure'),
              [Input('month-filter-selector', 'value'),
               Input('month-sensordataselector', 'value')])

def update_monthly_timeseries(selected_filters, selected_dropdown_value):
    ''' lineplot group by date '''
    if len(selected_dropdown_value) > 2:
        selected_dropdown_value = selected_dropdown_value[:2]

    if not isinstance(selected_filters, list):
        selected_filters = [selected_filters]

    # create fig object
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # add 'all' trace
    if 'all' in selected_filters:
        for data in selected_dropdown_value:
            data_daily_all = df.groupby('date', as_index=False).agg({data: "mean"})
            fig.add_trace(
                go.Scatter(x=data_daily_all["date"],
                           y=data_daily_all[data],
                           mode='lines',
                           opacity=0.7,
                           name=f'{data} (alle Räume)',
                           textposition='bottom center'),
                secondary_y=False if data == selected_dropdown_value[0] else True)

    # add traces for rooms
    
    for room in selected_filters:
        room_data = df[df['room_number'] == room]
        for data in selected_dropdown_value:
            room_daily = room_data.groupby('date', as_index=False).agg({data: "mean"})
            
            trace_name = f'{data} (alle Räume)' if room == 'all' else f'{data} (Raum {room})'
            fig.add_trace(
                go.Scatter(x=room_daily["date"],
                        y=room_daily[data],
                        mode='lines',
                        opacity=0.7,
                        name=trace_name,
                        textposition='bottom center'),
                secondary_y=False if data == selected_dropdown_value[0] else True)
                

    # update plot
    fig.update_layout(
        colorway=["#5E0DAC", '#FF4F00', '#375CB1', '#FF7400', '#FFF400', '#FF0056'],
        template='plotly_dark',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        plot_bgcolor='rgba(0, 0, 0, 0)',
        margin={'b': 15},
        hovermode='x',
        autosize=True,
        title={'text': 'Monatsübersicht', 'font': {'color': 'white'}, 'x': 0.5},
        xaxis={'title': 'Datum', 'range': [df.date.min(), df.date.max()]},           
    )
    return fig

@app.callback(Output('count-timeseries', 'figure'),
              [Input('count-filter-selector', 'value')])

def update_count_timeseries(selected_filters):
    ''' histogram per date '''

    
    if not isinstance(selected_filters, list):
        selected_filters = [selected_filters]

    # create fig object
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # add trace for 'all'
    if 'all' in selected_filters:
         fig.add_trace(
            go.Histogram(x=df["date"],
                        opacity=0.7,
                        name=f'count (alle Räume)'))

    #add traces for rooms
    
    for room in selected_filters:
        if room != 'all':
            room_data = df[df['room_number'] == room]
            trace_name = f'count (alle Räume)' if room == 'all' else f'count (Raum {room})'
            fig.add_trace(
                go.Histogram(x=room_data["date"],
                        opacity=0.7,
                        name=trace_name))
                

    # update layout of plot
    fig.update_layout(
        colorway=["#5E0DAC", '#FF4F00', '#375CB1', '#FF7400', '#FFF400', '#FF0056'],
        template='plotly_dark',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        plot_bgcolor='rgba(0, 0, 0, 0)',
        margin={'b': 15},
        hovermode='x',
        autosize=True,
        title={'text': 'Anzahl der gesammelten Daten über das Jahr verteilt', 'font': {'color': 'white'}, 'x': 0.5},
        xaxis={'title': 'Datum', 'range': [df.date.min(), df.date.max()]},           
    )
    return fig

if __name__ == '__main__':
    app.run_server(debug=True, dev_tools_ui = False)
