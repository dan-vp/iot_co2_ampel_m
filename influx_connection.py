import pandas as pd
import influxdb_client
from influxdb_client import Point
from influxdb_client.client.write_api import SYNCHRONOUS

# Secrets
token = "uTG-Rt2pM9JKTIZGr6ZYgRXcOoaG0wVHXn3ERZsmsdMdQE5wByhR7VSxcVIsATwrfQ8MvRzZ7Se4KdvrBEjHyQ=="
url = "https://iwi-i-influx-db-01.hs-karlsruhe.de:8086"
bucket = "iot_gebaeude_m"
org = "Vorlesung"

# Client erstellen und Read/Write API anlegen
write_client = influxdb_client.InfluxDBClient(url=url, token=token, org=org)
write_api = write_client.write_api(write_options=SYNCHRONOUS)
query_api = write_client.query_api()

# Daten lesen
# NOTE: hier wird nur eine Datei von vielen eingelesen!
df_m = pd.read_parquet("hka_M_preprocessed.parquet")

# Daten erstellen
# NOTE: Das muss natürlich nur einmal gemacht werden!
for index, row in df_m.iterrows():
    print(f"> ({(index+1):02}/{len(df_m)-1}) Adding Data Point ...")
    for key, value in row.to_dict().items():
        point = Point(f"{index}_{key}")
        point.tag("sensor", f"{index}")
        point.field("value", value)
        write_api.write(bucket=bucket, org=org, record=point)

# Query erstellen
query = f"""from(bucket: "{bucket}")
 |> range(start: -10m)"""

# Query ausführen
tables = query_api.query(query, org=org)

# Ergebnis ausgeben
for table in tables:
    for record in table.records:
        print(record)
