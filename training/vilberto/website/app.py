from flask import Flask, render_template
from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px
import pandas as pd

server = Flask(__name__)

app = Dash(__name__, server=server, url_base_pathname='/dash/')

df = pd.read_csv('../code/data_processed.csv')
df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")

app.layout = html.Div(children=[
    html.H1(children='Visualización de Datos Financieros de Walmart', style={'textAlign': 'center'}),

    html.Div([
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
            clearable=False,
            style={'width': '200px'}
        ),
      ], style={'display': 'flex', 'gap': '0.5rem', 'align-items':'center'}),

      html.Div([
        html.Label('Seleccionar Rango de Fechas:'),
        dcc.DatePickerRange(
            id='date-picker-range',
            start_date=df['Date'].min(),
            end_date=df['Date'].max(),
            display_format='YYYY-MM-DD',
        ),
      ], style={'display': 'flex', 'gap':'0.5rem', 'align-items':'center','float': 'right'}),

    ], style={
      'display':'flex',
      'align-items':'center',
      'justify-content':'space-around',
    }),

    html.Div(
      dcc.Graph(
        id='price-chart',
        style={
          'width':'100%'
        }
      ),
      style={
        'border-radius':'8px',
        'margin': '1.2rem 0.5rem',
      }
    ),

    html.A("Regresar", href="/", style={
      'padding': '0.3rem 0.6rem',
      'border-radius': '4px',
      'border': '1px solid #ccc',
      'text-decoration': 'none',
      'color': '#212121',
      'width': '100px',
      'text-align': 'center',
      'align-self': 'center',
    })
], style={
  'display': 'flex',
  'flex-direction': 'column',
  'gap': '1rem',
  'margin-top':'0.8rem'
})

@app.callback(
    Output('price-chart', 'figure'),
    Input('price-type-dropdown', 'value'),
    Input('date-picker-range', 'start_date'),
    Input('date-picker-range', 'end_date')
)
def update_graph(selected_price, start_date, end_date):
    filtered_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

    formatted_start_date = pd.to_datetime(start_date).strftime('%Y-%m-%d')
    formatted_end_date = pd.to_datetime(end_date).strftime('%Y-%m-%d')

    fig = px.line(
        filtered_df,
        x='Date',
        y=selected_price,
        title=f'{selected_price}: {formatted_start_date} hasta {formatted_end_date}'
    )

    fig.update_layout(xaxis_title='Fecha', yaxis_title='Precio', template='plotly_dark')

    return fig

@server.route('/')
def home():
    return render_template('index.html')

@server.route('/dash/')
def dash_app():
    return app.index()

if __name__ == '__main__':
    server.run(debug=True)