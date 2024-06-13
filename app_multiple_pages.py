import dash
import dash_html_components as html
import dash_core_components as dcc
# from dash import html, dcc
import plotly.graph_objects as go
import pandas as pd
from dash.dependencies import Input, Output
from plotly.subplots import make_subplots
import dash_bootstrap_components as dbc

# Load data
df = pd.read_parquet("../Gebäude_M_Analysis/Preprocessed_M_Data.parquet")
df["date"] = df["date_time"].dt.date
daily_counts = df.groupby('date').size().reset_index(name='count')

sensor_data_liste = ['tmp', 'hum', 'CO2', 'VOC', 'vis', 'IR']

app = dash.Dash(__name__)
app.config.suppress_callback_exceptions = True

def get_options(sensor_data):
    dict_list = []
    for i in sensor_data:
        dict_list.append({'label': i, 'value': i})

    return dict_list

options = get_options(sensor_data_liste)

sidebar = html.Div(
    [
        html.H1(f"Main Header", style={"fontSize": "36px", "fontWeight":"bold"}),
        html.Hr(),
        html.H2(f"Sub Header", style={"fontSize": "28px", "fontWeight":"bold"}),
        html.Hr(),
        dbc.Nav(
            [
                dbc.NavLink("Home", href="/", active="exact"),
                dbc.NavLink("Gesamtübersicht", href="/gesamtuebersicht", active="exact"),
                dbc.NavLink("Tagesübersicht", href="/tagesuebersicht", active="exact"),
            ],
    
            vertical=True,
            pills=True
        ),
    ],
    
)

app = dash.Dash(__name__)
content = html.Div(id="page-content")
app.layout = html.Div([dcc.Location(id="url"), sidebar, content])

@app.callback(Output("page-content", "children"), [Input("url", "pathname")])

def render_page_content(pathname):

    if pathname == f"/":
        return html.Div(
            html.H1("This is the home page.")
        )
    
    elif pathname == f"/gesamtuebersicht":
        return html.Div(
            html.H1("Das ist die Gesamtuebersicht.")
        )
    
    elif pathname == f"/tagesuebersicht":
        return html.Div([
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
if __name__ == '__main__':
    app.run_server(debug=True)