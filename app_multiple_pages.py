import dash
# import dash_html_components as html
# import dash_core_components as dcc
from dash import html, dcc
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pandas as pd
from plotly.subplots import make_subplots
from Preprocessing_M import DataExtractor, DataPreprocessing
from nbimporter import NotebookLoader
import numpy as np

# Load data
# df = pd.read_parquet("../Gebäude_M_Analysis/Preprocessed_M_Data.parquet")
df = pd.read_parquet("dashboard_data/eda_lenni.parquet")

loader = NotebookLoader()
notebook = loader.load_module('exploratory_data_analysis_dashboard')
remove_outliers = notebook.remove_outliers
resample = notebook.resample
scale = notebook.scale
missing_dates = notebook.missing_dates

df["date"] = df["date_time"].dt.date
df["hour"] = df["date_time"].dt.hour
daily_counts = df.groupby('date').size().reset_index(name='count')

sensor_data_liste = ['tmp', 'hum', 'CO2', 'VOC', 'vis', 'IR']
room_numbers = df['room_number'].unique()

# Initialize the app
app = dash.Dash(__name__)
app.config.suppress_callback_exceptions = True

def get_options(sensor_data):
    return [{'label': i, 'value': i} for i in sensor_data]

options = get_options(sensor_data_liste)
room_options = get_options(room_numbers)

# Layout for the sidebar
sidebar_layout = html.Div(
    children=[
        html.H2('Navigation'),
        html.Ul(
            children=[
                html.Li(dcc.Link('Gesamtübersicht', href='/')),
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
        dcc.Location(id='url', refresh=False),
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
        return gesamtuebersicht_layout

all_days, missing_days = missing_dates(df, "date_time", "D")
missing_days_plotly = [day.strftime("%Y-%m-%d") for day in missing_days]

weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
sum_per_weekday = {k: 0 for k in weekdays}
for day in missing_days:
    sum_per_weekday[weekdays[day.weekday()]] += 1

all_hours, missing_hours = missing_dates(df, date_col="date_time", freq='h')
sum_per_hour = {k:0 for k in range(24)}
for hour in missing_hours:
    sum_per_hour[hour.hour] += 1

gesamtuebersicht_layout = html.Div(
    children=[
        html.H1('Gesamtübersicht', style={'color': 'white'}),

        # Plot für fehlende Tage über die Zeit
        dcc.Graph(
            id='missing-dates-plot',
            figure={
                'data': [
                    {'x': all_days, 'y': np.ones(len(all_days)), 'type': 'scatter', 'mode': 'lines', 'line': {'color': '#636EFA'}, 'name': 'Available days'},
                    {'x': missing_days, 'y': np.ones(len(missing_days)), 'type': 'scatter', 'mode': 'markers', 'marker': {'size': 10, 'color': '#EF553B'}, 'name': 'Missing days'},
                ],
                'layout': {
                    'title': 'Missing days over time',
                    'xaxis': {'title': 'Date', 'titlefont': {'color': 'white'}, 'tickfont': {'color': 'white'}},
                    'yaxis': {'title': 'Available', 'titlefont': {'color': 'white'}, 'tickfont': {'color': 'white'}},
                    'legend': {'x': 0.8, 'y': 0.95, 'traceorder': 'normal', 'bgcolor': '#1E1E1E', 'bordercolor': 'white', 'borderwidth': 2, 'font': {'color': 'white'}},
                    'margin': {'l': 40, 'b': 40, 't': 40, 'r': 40},
                    'template': 'plotly_dark',
                    'plot_bgcolor': '#1E1E1E',
                    'paper_bgcolor': '#1E1E1E',
                    'font': {'color': 'white'}
                }
            }
        ),

        # Untere Ebene für die Barplots
        html.Div(
            children=[
                # Plot für fehlende Einträge pro Wochentag
                dcc.Graph(
                    id='missing-entries-per-weekday',
                    figure={
                        'data': [
                            {'x': list(sum_per_weekday.keys()), 'y': list(sum_per_weekday.values()), 'type': 'bar', 'name': 'Missing entries', 'marker': {'color': '#EF553B'}}
                        ],
                        'layout': {
                            'title': 'Number of missing entries per weekday',
                            'xaxis': {'title': 'Weekday', 'titlefont': {'color': 'white'}, 'tickfont': {'color': 'white'}},
                            'yaxis': {'title': 'Missing entries', 'titlefont': {'color': 'white'}, 'tickfont': {'color': 'white'}},
                            'margin': {'l': 40, 'b': 40, 't': 40, 'r': 40},
                            'template': 'plotly_dark',
                            'plot_bgcolor': '#1E1E1E',
                            'paper_bgcolor': '#1E1E1E',
                            'font': {'color': 'white'},
                            'height': 250  # Höhe des Plots auf 250px setzen
                        }
                    }
                ),

                # Plot für fehlende Einträge pro Stunde
                dcc.Graph(
                    id='missing-entries-per-hour',
                    figure={
                        'data': [
                            {'x': list(sum_per_hour.keys()), 'y': list(sum_per_hour.values()), 'type': 'bar', 'name': 'Missing entries', 'marker': {'color': '#EF553B'}}
                        ],
                        'layout': {
                            'title': 'Number of missing entries per hour',
                            'xaxis': {'title': 'Hour', 'titlefont': {'color': 'white'}, 'tickfont': {'color': 'white'}},
                            'yaxis': {'title': 'Missing entries', 'titlefont': {'color': 'white'}, 'tickfont': {'color': 'white'}},
                            'margin': {'l': 40, 'b': 40, 't': 40, 'r': 40},
                            'template': 'plotly_dark',
                            'plot_bgcolor': '#1E1E1E',
                            'paper_bgcolor': '#1E1E1E',
                            'font': {'color': 'white'},
                            'height': 250  # Höhe des Plots auf 250px setzen
                        }
                    }
                )
            ],
            style={'display': 'flex', 'flex-direction': 'row', 'justify-content': 'space-between', 'width': '100%', 'margin-top': '20px'}
        )
    ],
    style={'backgroundColor': '#1E1E1E', 'color': 'white', 'padding': '20px'}
)
                  

# # Layout for Gesamtübersicht
# gesamtuebersicht_layout = html.Div(
#     children=[
#         html.H1('Gesamtübersicht'),
#         html.P('Diese Seite ist noch in Bearbeitung.')
#     ]
# )

# Layout for Monatsübersicht
monatsuebersicht_layout = html.Div(
    children=[
        html.H1('Monatsübersicht'),
        html.P('Wähle ein oder zwei Sensordaten, die du visualisieren willst'),
        dcc.Dropdown(id='month-sensordataselector', options=options,
                     multi=True, value=[sensor_data_liste[0]],
                     style={'backgroundColor': '#1E1E1E'},
                     className='sensordataselector'
        ),
        html.Div(id='month-input-warning'),
        dcc.Graph(id='month-timeseries', config={'displayModeBar': False}, animate=True),
        html.P('Wähle eine Raumnummer und Sensordaten, die du visualisieren willst'),
        dcc.Dropdown(id='month-raumselector', options=room_options,
                     multi=False, value=room_numbers[0],
                     style={'backgroundColor': '#1E1E1E'},
                     className='raumselector'
        ),
        dcc.Dropdown(id='month-raum-sensordataselector', options=options,
                     multi=True, value=[sensor_data_liste[0]],
                     style={'backgroundColor': '#1E1E1E'},
                     className='sensordataselector'
        ),
        dcc.Graph(id='month-raum-timeseries', config={'displayModeBar': False}, animate=True)
    ]
)

# Layout for Tagesübersicht
tagesuebersicht_layout = html.Div(
    children=[
        html.H1('Tagesübersicht'),
        html.P('Wähle ein oder zwei Sensordaten, die du visualisieren willst'),
        dcc.Dropdown(id='sensordataselector', options=options,
                     multi=True, value=[sensor_data_liste[0]],
                     style={'backgroundColor': '#1E1E1E'},
                     className='sensordataselector'
        ),
        html.Div(id='input-warning'),
        dcc.Graph(id='timeseries', config={'displayModeBar': False}, animate=True),
        html.P('Wähle eine Raumnummer und Sensordaten, die du visualisieren willst'),
        dcc.Dropdown(id='raumselector', options=room_options,
                     multi=False, value=room_numbers[0],
                     style={'backgroundColor': '#1E1E1E'},
                     className='raumselector'
        ),
        dcc.Dropdown(id='raum-sensordataselector', options=options,
                     multi=True, value=[sensor_data_liste[0]],
                     style={'backgroundColor': '#1E1E1E'},
                     className='sensordataselector'
        ),
        dcc.Graph(id='raum-timeseries', config={'displayModeBar': False}, animate=True)
    ]
)

forecast_layout = html.Div(
    children=[
        html.H1('Forecast'),
        html.P('Diese Seite ist noch in Bearbeitung.')
    ]
)

@app.callback(
    [Output("input-warning", 'children'), Output("sensordataselector", "options")], 
    [Input("sensordataselector", "value")]
)
def limit_drop_options(symbols):
    """Limit dropdown to at most two active selections"""
    if len(symbols) > 1:
        warning = html.P("You have entered the limit", id='input-warning')
        return [
            warning,
            [option for option in options if option["value"] in symbols]
        ]
    else:
        return [[], options]

@app.callback(Output('timeseries', 'figure'),
              [Input('sensordataselector', 'value')])
def update_timeseries(selected_dropdown_value):
    ''' Draw traces of the feature 'value' based one the currently selected stocks '''
    if len(selected_dropdown_value) > 2:
        selected_dropdown_value = selected_dropdown_value[:2]

    df_sub = df
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    for data in selected_dropdown_value:
        CO2_daily = df_sub.groupby('hour', as_index=False).agg({data: "mean"})
        fig.add_trace(
            go.Scatter(x=CO2_daily["hour"],
                       y=CO2_daily[data],
                       mode='lines',
                       opacity=0.7,
                       name=data,
                       textposition='bottom center'),
            secondary_y=False if data == selected_dropdown_value[0] else True)

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
    [Output("month-input-warning", 'children'), Output("month-sensordataselector", "options")], 
    [Input("month-sensordataselector", "value")]
)
def limit_month_drop_options(symbols):
    """Limit dropdown to at most two active selections"""
    if len(symbols) > 1:
        warning = html.P("You have entered the limit", id='month-input-warning')
        return [
            warning,
            [option for option in options if option["value"] in symbols]
        ]
    else:
        return [[], options]

@app.callback(Output('month-timeseries', 'figure'),
              [Input('month-sensordataselector', 'value')])
def update_month_timeseries(selected_dropdown_value):
    ''' Draw traces of the feature 'value' based on the currently selected stocks grouped by date '''
    if len(selected_dropdown_value) > 2:
        selected_dropdown_value = selected_dropdown_value[:2]

    df_sub = df
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    for data in selected_dropdown_value:
        data_daily = df_sub.groupby('date', as_index=False).agg({data: "mean"})
        fig.add_trace(
            go.Scatter(x=data_daily["date"],
                       y=data_daily[data],
                       mode='lines',
                       opacity=0.7,
                       name=data,
                       textposition='bottom center'),
            secondary_y=False if data == selected_dropdown_value[0] else True)

    fig.update_layout(
        colorway=["#5E0DAC", '#FF4F00', '#375CB1', '#FF7400', '#FFF400', '#FF0056'],
        template='plotly_dark',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        plot_bgcolor='rgba(0, 0, 0, 0)',
        margin={'b': 15},
        hovermode='x',
        autosize=True,
        title={'text': 'Monatsübersicht', 'font': {'color': 'white'}, 'x': 0.5},
    )

    return fig

@app.callback(
    Output('raum-timeseries', 'figure'),
    [Input('raumselector', 'value'),
     Input('raum-sensordataselector', 'value')]
)
def update_raum_timeseries(selected_raum, selected_dropdown_value):
    ''' Draw traces of the feature 'value' based on the currently selected room and sensor data '''
    if len(selected_dropdown_value) > 2:
        selected_dropdown_value = selected_dropdown_value[:2]

    df_sub = df[df['room_number'] == selected_raum]
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    for data in selected_dropdown_value:
        CO2_daily = df_sub.groupby('hour', as_index=False).agg({data: "mean"})
        fig.add_trace(
            go.Scatter(x=CO2_daily["hour"],
                       y=CO2_daily[data],
                       mode='lines',
                       opacity=0.7,
                       name=data,
                       textposition='bottom center'),
            secondary_y=False if data == selected_dropdown_value[0] else True)

    fig.update_layout(
        colorway=["#5E0DAC", '#FF4F00', '#375CB1', '#FF7400', '#FFF400', '#FF0056'],
        template='plotly_dark',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        plot_bgcolor='rgba(0, 0, 0, 0)',
        margin={'b': 15},
        hovermode='x',
        autosize=False,
        title={'text': 'Tagesübersicht für Raum {}'.format(selected_raum), 'font': {'color': 'white'}, 'x': 0.5},
    )

    return fig

@app.callback(
    Output('month-raum-timeseries', 'figure'),
    [Input('month-raumselector', 'value'),
     Input('month-raum-sensordataselector', 'value')]
)
def update_month_raum_timeseries(selected_raum, selected_dropdown_value):
    ''' Draw traces of the feature 'value' based on the currently selected room and sensor data grouped by date '''
    if len(selected_dropdown_value) > 2:
        selected_dropdown_value = selected_dropdown_value[:2]

    df_sub = df[df['room_number'] == selected_raum]
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    for data in selected_dropdown_value:
        data_daily = df_sub.groupby('date', as_index=False).agg({data: "mean"})
        fig.add_trace(
            go.Scatter(x=data_daily["date"],
                       y=data_daily[data],
                       mode='lines',
                       opacity=0.7,
                       name=data,
                       textposition='bottom center'),
            secondary_y=False if data == selected_dropdown_value[0] else True)

    fig.update_layout(
        colorway=["#5E0DAC", '#FF4F00', '#375CB1', '#FF7400', '#FFF400', '#FF0056'],
        template='plotly_dark',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        plot_bgcolor='rgba(0, 0, 0, 0)',
        margin={'b': 15},
        hovermode='x',
        autosize=True,
        title={'text': 'Monatsübersicht für Raum {}'.format(selected_raum), 'font': {'color': 'white'}, 'x': 0.5},
    )

    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
