from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px
import pandas as pd

# Crear la aplicación Flask
server = Flask(__name__)

# Crear la aplicación Dash dentro de Flask
app = Dash(__name__, server=server, url_base_pathname='/dash/')

df = pd.read_csv('../code/data_processed.csv')

df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")

app = Dash(__name__)

app.layout = html.Div(children=[
    html.H1(children='Visualización de Datos Financieros de Walmart', style={'textAlign': 'center'}),

    html.Div([
        html.Label('Seleccionar Tipo de Precio:'),
        dcc.Dropdown(
            id='price-type-dropdown',
            options=[
                {'label': 'Apertura', 'value': 'Open'},
                {'label': 'Cierre', 'value': 'Close'},
                {'label': 'Máximo', 'value': 'High'},
                {'label': 'Mínimo', 'value': 'Low'},
                {'label': 'Cierre Ajustado', 'value': 'Adj Close'},
            ],
            value='Close',
            clearable=False
        ),
    ], style={'width': '48%', 'display': 'flex'}),

    html.Div([
        html.Label('Seleccionar Rango de Fechas:'),
        dcc.DatePickerRange(
            id='date-picker-range',
            start_date=df['Date'].min(),
            end_date=df['Date'].max(),
            display_format='YYYY-MM-DD'
        ),
    ], style={'width': '48%', 'display': 'inline-block', 'float': 'right'}),

    dcc.Graph(
        id='price-chart'
    )
])

@app.callback(
    Output('price-chart', 'figure'),
    Input('price-type-dropdown', 'value'),
    Input('date-picker-range', 'start_date'),
    Input('date-picker-range', 'end_date')
)
def update_graph(selected_price, start_date, end_date):
    filtered_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

    fig = px.line(
        filtered_df,
        x='Date',
        y=selected_price,
        title=f'Precio {selected_price} desde {start_date} hasta {end_date}'
    )

    fig.update_layout(xaxis_title='Fecha', yaxis_title='Precio', template='plotly_dark')

    return fig