from flask import Flask, render_template
from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px
import pandas as pd
from sklearn.model_selection import train_test_split
import plotly.graph_objs as go
import pickle
import joblib

server = Flask(__name__)

app = Dash(__name__, server=server, url_base_pathname='/dash/')

df = pd.read_csv('./static/data-processed/Dataset_Cleaned.csv')
df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%y")


# Cargar el modelo entrenado desde el archivo .pkl
model = joblib.load("static/data-processed/Gold_Price_Prediction_Model.pkl")

# Cargar los datos nuevamente (los mismos utilizados para el entrenamiento)
file_path = "static/data-processed/Selected_Features_Gold_Price.csv"
data = pd.read_csv(file_path)

# Definir las características (X) y el objetivo (y) nuevamente
X = data.drop(columns=['Price Tomorrow'])
y = data['Price Tomorrow']

# Realizar predicciones con el modelo entrenado
y_pred = model.predict(X)


app.layout = html.Div([
  html.H1(children='Visualización de los datos', style={ "margin":"0 0 0.8rem 0.8rem"}),
  html.Div([
    html.H2("Gráfico del dataset", style={"text-align":"center"}),
    html.Div([
      html.Div([
        html.Label('Seleccionar Encabezado:'),
        dcc.Dropdown(
          id="list_data_dropdown",
          options=[
              {'label': 'Precio 2 días antes', 'value': 'Price 2 Days Prior'},
              {'label': 'Precio 1 día antes', 'value': 'Price 1 Day Prior'},
              {'label': 'Precio hoy', 'value': 'Price Today'},
              {'label': 'Precio mañana', 'value': 'Price Tomorrow'},
              {'label': 'Cambio de precio mañana', 'value': 'Price Change Tomorrow'},
              {'label': 'Cambio de precio dentro de 10 días', 'value': 'Price Change Ten'},
              {'label': 'Desviación estándar en los últimos 10 días', 'value': 'Std Dev 10'},
              {'label': 'Media móvil de 20 días', 'value': 'Twenty Moving Average'},
              {'label': 'Media móvil de 50 días', 'value': 'Fifty Day Moving Average'},
              {'label': 'Media móvil de 200 días', 'value': '200 Day Moving Average'},
              {'label': 'Tasa de inflación mensual', 'value': 'Monthly Inflation Rate'},
              {'label': 'Tasa EFFR', 'value': 'EFFR Rate'},
              {'label': 'Volumen', 'value': 'Volume'},
              {'label': 'Rendimiento par del Tesoro a un mes', 'value': 'Treasury Par Yield Month'},
              {'label': 'Rendimiento par del Tesoro a dos años', 'value': 'Treasury Par Yield Two Year'},
              {'label': 'Rendimiento par del Tesoro a 10 años', 'value': 'Treasury Par Yield Curve Rates (10 Yr)'},
              {'label': 'Índice DXY', 'value': 'DXY'},
              {'label': 'Apertura del S&P', 'value': 'SP Open'},
              {'label': 'VIX', 'value': 'VIX'},
              {'label': 'Petróleo crudo', 'value': 'Crude'},
          ],
          value='Price Today',
          clearable=False,
          style={'width': '300px'}
        ),
      ], style={'display': 'flex', 'gap': '0.5rem', 'align-items':'center'}),

      html.Div([
        html.Label('Seleccionar Rango de Fechas:'),
        dcc.DatePickerRange(
          id="date_picker",
          start_date=df['Date'].min(),
          end_date=df['Date'].max(),
          display_format='YYYY-MM-DD',
          className="date-picker-range",
        ),
      ], style={'display': 'flex', 'gap':'0.5rem', 'align-items':'center','float': 'right'}),

    ], className="container-inputs"),

    html.Div([
      dcc.Graph(id="list_data"),
    ], style={'display': 'flex', 'flex-direction': 'column', 'gap': '1.2rem', 'margin-top': '0.8rem'}),
  ], style={"margin": "0 0.8rem", "padding":"0.8rem", "border":"1px solid #ccc", "border-radius":"4px"}),

  html.Div([
    html.H2("Modelo de predicción usando Regresión lineal", style={"text-align":"center","margin-bottom":"1rem"}),

    html.Div([
      html.Div([
        html.Label("Seleccionar:"),

        dcc.Dropdown(
          id="graph-type-dropdown",
          options=[
            {'label': 'Valores Reales vs Predicciones', 'value': 'real_vs_pred'},
            {'label': 'Distribución de Errores', 'value': 'error_distribution'}
          ],
          value='real_vs_pred',
          style={"width": "340px"}
        ),

      ], style={"display":"flex", "gap":"0.8rem", "align-items": "center"}),

      dcc.Graph(
        id="graph"
      )

    ], style={'display': 'flex', 'flex-direction': 'column', 'gap': '1.5rem'}),

  ], style={"margin": "0 0.8rem", "border": "1px solid #ccc", "border-radius":"4px", "padding":"0.8rem"}),

  html.Div([
    html.H2("Matriz de Correlación", style={"text-align":"center", "margin-bottom":"0.5rem"}),

    html.Div([
        dcc.Checklist(
          id="medals",
          options=["Std Dev 10", "Twenty Moving Average", "Monthly Inflation Rate", "EFFR Rate", "Treasury Par Yield Month","Treasury Par Yield Two Year","Treasury Par Yield Curve Rates (10 Yr)","DXY","SP Open","VIX", "Crude"],
          value=["Std Dev 10"],
          className="checkbox"
        ),
    ]),

    html.Div([
      dcc.Graph(id="graph-correlation")
    ], style={"display":"flex", "align-items":"center", "justify-content":"center", "margin-top":"1.2rem"}),

  ], style={"border": "1px solid #ccc", "border-radius":"4px", "margin": "0 0.8rem", "padding":"0.8rem"}),

  html.Div([
    html.H2("Gráfica de dispersión", style={"text-align":"center", "margin-bottom":"1.2rem"}),

    html.Div([
      html.Label("Seleccionar: "),

      dcc.Dropdown(
        id="dropdown-scatter",
        options=[{'label': col, 'value': col} for col in data.columns],
        value=data.columns[0],
        style={"width":"300px"}
      ),
    ], style={"display":"flex", "gap":"0.8rem", "align-items":"center"}),

    dcc.Graph(
      id="graph-scatter",
    )

  ], style={"margin": "0 0.8rem", "border": "1px solid #ccc", "border-radius":"4px", "padding":"0.8rem"}),

  html.A("Regresar", href="/", className="back-btn")

], style={
    'display': 'flex',
    'flex-direction': 'column',
    'gap': '1rem',
    'margin-top':'0.8rem'
})

# Actualización en la visualización de los datos del dataset
@app.callback(
    Output('list_data', 'figure'),
    Input('list_data_dropdown', 'value'),
    Input('date_picker', 'start_date'),
    Input('date_picker', 'end_date'),
)
def update_list_data_graph(selected_option, start_date, end_date):
    filtered_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

    formatted_start_date = pd.to_datetime(start_date).strftime('%Y-%m-%d')
    formatted_end_date = pd.to_datetime(end_date).strftime('%Y-%m-%d')

    fig = px.line(
        filtered_df,
        x='Date',
        y=selected_option,
        title=f'{selected_option}: {formatted_start_date} hasta {formatted_end_date}'
    )

    fig.update_layout(xaxis_title='Fecha', yaxis_title='Precio')

    return fig

# Actualización en la visualización de los datos del dataset vs el modelo predictivo
@app.callback(
  Output('graph', 'figure'),
  [Input("graph-type-dropdown", 'value')]
)
def update_graph(selected_graph_type): 
    if selected_graph_type == 'real_vs_pred':
        # Gráfico de líneas para valores reales vs predicciones
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(len(y))), y=y, mode='lines', name='Valores Reales'))
        fig.add_trace(go.Scatter(x=list(range(len(y_pred))), y=y_pred, mode='lines', name='Predicciones'))
        fig.update_layout(
            title='Valores Reales vs Predicciones',
            xaxis_title='Índice de la Muestra',
            yaxis_title='Precio del Oro'
        )
    elif selected_graph_type == 'error_distribution':
        # Calcular los errores (diferencia entre valores reales y predicciones)
        errors = y - y_pred
        fig = go.Figure(data=[go.Histogram(x=errors)])
        fig.update_layout(
            title='Distribución de Errores',
            xaxis_title='Error (Valor Real - Predicción)',
            yaxis_title='Frecuencia'
        )
    return fig

# Actualización en la visualización de la matriz correlacional
@app.callback(
  Output('graph-correlation','figure'),
  Input('medals','value')
)
def filter_heatmap(cols):
  correlation_matrix = df[cols].corr()

  fig = px.imshow(
      correlation_matrix,
      text_auto=True,
      color_continuous_scale='RdBu_r',
      zmin=-1, zmax=1,
      labels=dict(x="Variables", y="Variables", color="Correlación"),
  )

  fig.update_layout(
      xaxis={'side': 'bottom'},
      yaxis_autorange='reversed',
      width=800, height=600,
      margin=dict(l=40, r=40, t=40, b=40) 
  )

  return fig

# Actualización en la visualización del gráfico de dispersión
@app.callback(
  Output("graph-scatter", "figure"),
  [Input("dropdown-scatter", "value")]
)
def upgrade_graph_scatter(x_axis):
  fig = px.scatter(
    data_frame=data,
    x=x_axis,
    y='Price Tomorrow',  # Variable dependiente
    title=f'Dispersión de Precio del Oro vs {x_axis}',
    labels={x_axis: x_axis, 'Price Tomorrow': 'Precio del Oro Mañana'}
  )

  return fig

# Rutas
@server.route('/')
def home():
    return render_template('index.html')

@server.route('/dash/')
def dash_app():
    return app.index()

if __name__ == '__main__':
    server.run(debug=True)