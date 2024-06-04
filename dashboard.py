# Zuerst müssen die benötigten Bibliotheken importiert werden
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd

# Beispiel-Datensatz laden (hier: aus der Parquet-Datei)
df_m = pd.read_parquet("Preprocessed_M_Data.parquet")

# Daten für den ersten Plot vorbereiten
df_m["date"] = df_m["date_time"].dt.date
daily_counts = df_m.groupby('date').size().reset_index(name='count')

# Daten für den zweiten Plot vorbereiten
m_building_rooms = df_m.room_number.unique()
num_rooms = len(m_building_rooms)
cols = 4
rows = -(- num_rooms // cols)  # ceiling division

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

fig_subplots.update_layout(height=300*rows, width=1800, title_text="Daily Measurement Counts per Room", showlegend=False)
fig_subplots.update_xaxes(title_text="Date")
fig_subplots.update_yaxes(title_text="Count")
# Layout für die Dash-Anwendung definieren
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Visualisierungen"),
    
    # Erster Plot: Linienplot
    dcc.Graph(
        id='line-plot',
        figure=px.line(daily_counts, x="date", y="count")
    ),
    
    # Zweiter Plot: Subplots für jeden Raum
    dcc.Graph(
        id='room-subplots',
        figure=fig_subplots
    )
])

# App starten
if __name__ == '__main__':
    app.run_server(debug=True)
