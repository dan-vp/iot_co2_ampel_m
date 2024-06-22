import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd

# Beispiel-Datensatz laden (hier: aus der Parquet-Datei)
df_m = pd.read_parquet("../Preprocessed_M_Data.parquet")
df_m.reset_index(drop=True, inplace=True)

# Daten für den ersten Plot vorbereiten
df_m["date"] = df_m["date_time"].dt.date
daily_counts = df_m.groupby('date').size().reset_index(name='count')

# Daten für den zweiten Plot vorbereiten
m_building_rooms = df_m.room_number.unique()
num_rooms = len(m_building_rooms)

rows = 2
cols = (num_rooms + rows - 1) // rows  # Anzahl der Spalten basierend auf 3 Zeilen


# Subplots für den zweiten Plot erstellen
fig_subplots = make_subplots(rows=rows, cols=cols, shared_xaxes=True, shared_yaxes=True,
                             subplot_titles=[f"Room {room}" for room in m_building_rooms])

# Schleife über jeden Raum
for i, room in enumerate(m_building_rooms, start=1):
    df_room = df_m[df_m.room_number == room]
    daily_count_room = df_room.groupby('date').size().reset_index(name='count')

    # Linie hinzufügen
    row = (i - 1) // cols + 1
    col = (i - 1) % cols + 1
    fig_subplots.add_trace(go.Scatter(x=daily_count_room['date'], y=daily_count_room['count'],
                                      mode='lines', line=dict(color='blue'), name=f"Room {room}"),
                           row=row, col=col)
    
    fig_subplots.update_xaxes(showticklabels=True, row=row, col=col)

# Layout anpassen

fig_subplots.update_layout(height=500, width=cols * 500, showlegend=False, paper_bgcolor="#D6EAF8")
start_date = df_m.date.min()
end_date = df_m.date.max()
fig_subplots.update_xaxes(range=[start_date, end_date])

# Dash-Anwendung definieren
app = dash.Dash(__name__, external_stylesheets=["./assets/style.css"])

app.layout = html.Div([
    html.Div([
    html.H2("Dashboard"),
    html.Details([
        html.Summary("Anzahl der Messungen"),
        html.Ul([
            html.Li(html.A("Insgesamt", href="#line-plot")),
            html.Li(html.A("Pro Raum", href="#room-subplots"))
        ])
    ], style={'background-color': 'black', 'padding': '10px', 'border': '1px solid #ddd', 'border-radius': '5px', 'margin-bottom': '10px'}),
], className="sidebar"),

    html.Div([
        html.H1("Visualisierungen"),
        
        html.Div([
            dcc.Graph(
                id='line-plot',
                figure=px.line(daily_counts, x="date", y="count", title="Anzahl der Messungen im Verlauf der Zeit").update_layout(height=500,paper_bgcolor="#D6EAF8")
            ),
        ], id="plot1"),
        
        html.Div([
            html.Div([
                dcc.Graph(
                    id='room-subplots',
                    figure=fig_subplots,
                    config={'displayModeBar': False},
                    style={'width': f'{cols * 500}px'}  # Breite des Graphen setzen
                ),
            ], className="scroll-content plot-container", style={'width': f'{cols * 500 + 20}px'}),  
        ], className="scroll-container", id="plot2")
    ], className="content")
])

# App starten
if __name__ == '__main__':
    app.run_server(debug=True)