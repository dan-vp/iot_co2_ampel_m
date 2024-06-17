import dash
import dash_html_components as html
import dash_core_components as dcc
# from dash import html, dcc
import plotly.graph_objects as go
import pandas as pd
from dash.dependencies import Input, Output
from plotly.subplots import make_subplots

# Load data
df = pd.read_parquet("../Gebäude_M_Analysis/Preprocessed_M_Data.parquet")
df["date"] = df["date_time"].dt.date
daily_counts = df.groupby('date').size().reset_index(name='count')

sensor_data_liste = ['tmp', 'hum', 'CO2', 'VOC', 'vis', 'IR']

# Initialize the app
app = dash.Dash(__name__)
app.config.suppress_callback_exceptions = True


def get_options(sensor_data):
    dict_list = []
    for i in sensor_data:
        dict_list.append({'label': i, 'value': i})

    return dict_list

options = get_options(sensor_data_liste)

app.layout = html.Div(
    children=[
        html.Div(className='row',
                 children=[
                    html.Div(className='four columns div-user-controls',
                             children=[
                                 html.H2('Gebäude M - Analysis'),
                                 html.P('Visualising time series with Plotly - Dash.'),
                                 html.P('Wähle ein oder zwei Sensordaten, die du visualisieren willst'),
                                 html.Div(
                                     className='div-for-dropdown',
                                     children=[
                                         dcc.Dropdown(id='sensordataselector', options=options,
                                                      multi=True, value=[sensor_data_liste[0]],
                                                      style={'backgroundColor': '#1E1E1E'},
                                                      className='sensordataselector'
                                                      ),
                                     ],
                                     style={'color': '#1E1E1E'}),
                                 html.Div(id='input-warning')
                                ]
                             ),
                    html.Div(className='eight columns div-for-charts bg-grey',
                             children=[
                                 dcc.Graph(id='timeseries',
                                     config={'displayModeBar': False},
                                     animate=True),
                                #  dcc.Graph(id='change',
                                #      config={'displayModeBar': False},
                                #      animate=True),
                             ])
                              ])
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

# Callback for timeseries price
@app.callback(Output('timeseries', 'figure'),
              [Input('sensordataselector', 'value')])
def update_timeseries(selected_dropdown_value):
    ''' Draw traces of the feature 'value' based one the currently selected stocks '''

    if len(selected_dropdown_value) > 2:
        selected_dropdown_value = selected_dropdown_value[:2]

    # STEP 1
    # trace = []
    df_sub = df
    # STEP 2
    # Draw and append traces for each stock

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    for data in selected_dropdown_value:

        CO2_daily = df_sub.groupby('hour', as_index=False).agg({data:"mean"})

        fig.add_trace(
            go.Scatter(x=CO2_daily["hour"],
                                 y=CO2_daily[data],
                                 mode='lines',
                                 opacity=0.7,
                                 name=data,
                                 textposition='bottom center'),
            secondary_y=False if data == selected_dropdown_value[0] else True)
        
    # Define Figure
    # STEP 4
    fig.update_layout(
                  colorway=["#5E0DAC", '#FF4F00', '#375CB1', '#FF7400', '#FFF400', '#FF0056'],
                  template='plotly_dark',
                  paper_bgcolor='rgba(0, 0, 0, 0)',
                  plot_bgcolor='rgba(0, 0, 0, 0)',
                  margin={'b': 15},
                  hovermode='x',
                  autosize=True,
                  title={'text': 'Tagesübersicht', 'font': {'color': 'white'}, 'x': 0.5},
                  #xaxis={'range': [df_sub.index.min(), df_sub.index.max()]},
              ),
              

    return fig


# @app.callback(Output('change', 'figure'),
#               [Input('stockselector', 'value')])
# def update_change(selected_dropdown_value):
#     ''' Draw traces of the feature 'change' based one the currently selected stocks '''
#     trace = []
#     df_sub = df
#     # Draw and append traces for each stock
#     for stock in selected_dropdown_value:
#         trace.append(go.Scatter(x=df_sub[df_sub['stock'] == stock].index,
#                                  y=df_sub[df_sub['stock'] == stock]['change'],
#                                  mode='lines',
#                                  opacity=0.7,
#                                  name=stock,
#                                  textposition='bottom center'))
#     traces = [trace]
#     data = [val for sublist in traces for val in sublist]
#     # Define Figure
#     figure = {'data': data,
#               'layout': go.Layout(
#                   colorway=["#5E0DAC", '#FF4F00', '#375CB1', '#FF7400', '#FFF400', '#FF0056'],
#                   template='plotly_dark',
#                   paper_bgcolor='rgba(0, 0, 0, 0)',
#                   plot_bgcolor='rgba(0, 0, 0, 0)',
#                   margin={'t': 50},
#                   height=250,
#                   hovermode='x',
#                   autosize=True,
#                   title={'text': 'Daily Change', 'font': {'color': 'white'}, 'x': 0.5},
#                   xaxis={'showticklabels': False, 'range': [df_sub.index.min(), df_sub.index.max()]},
#               ),
#               }

#     return figure


if __name__ == '__main__':
    app.run_server(debug=True)