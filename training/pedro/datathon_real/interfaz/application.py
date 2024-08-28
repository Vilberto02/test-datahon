import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import joblib
from datetime import datetime, timedelta
import plotly.graph_objs as go

# Cargar los datos y los modelos entrenados
data_path = "../seleccion_caracteristicas_relacionadas/Treated_Gold_Price_Data.csv"
data = pd.read_csv(data_path)
models = {
    "Linear Regression": joblib.load("modelos/Linear_Regression_Model.pkl"),
    "Decision Tree": joblib.load("modelos/Decision_Tree_Model.pkl"),
    "Random Forest": joblib.load("modelos/Random_Forest_Model.pkl"),
    "Support Vector Regression": joblib.load("modelos/SVR_Model.pkl"),
    "XGBoost": joblib.load("modelos/XGBoost_Model.pkl"),
}

# Obtener la última fecha registrada en los datos
data['Date'] = pd.to_datetime(data['Date'], format="%m%d%y", errors='coerce')
last_date = data['Date'].max()

# Crear la aplicación Dash
app = dash.Dash(__name__)

# Layout de la aplicación
app.layout = html.Div([
    html.H1("Predicción del Precio del Oro"),
    dcc.Dropdown(
        id='model-selector',
        options=[{'label': model, 'value': model} for model in models.keys()],
        value='Linear Regression',
        placeholder="Selecciona un modelo"
    ),
    html.Div([
        dcc.Graph(id='future-predictions-graph'),
    ]),
    html.Hr(),
    html.H2("Predicción para una Fecha Específica"),
    dcc.Input(
        id='date-input',
        type='text',
        placeholder="Ingresa una fecha (YYYY-MM-DD)",
    ),
    html.Button('Predecir', id='predict-button', n_clicks=0),
    html.Div(id='prediction-output')
])

# Callback para actualizar el gráfico con predicciones futuras
@app.callback(
    Output('future-predictions-graph', 'figure'),
    Input('model-selector', 'value')
)
def update_future_predictions(model_name):
    model = models[model_name]

    # Generar fechas futuras para predicción
    future_dates = pd.date_range(last_date + timedelta(days=1), periods=10).to_frame(index=False, name='Date')
    
    # Usar los valores más recientes como base para la predicción
    last_values = data.iloc[-1].drop(['Date', 'Price Tomorrow']).values.reshape(1, -1)
    predictions = []
    for _ in range(10):
        pred = model.predict(last_values)[0]
        predictions.append(pred)
        last_values = [pred]  # Simulación sencilla para mostrar en gráficos

    # Crear gráfico
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=future_dates['Date'], y=predictions, mode='lines+markers', name='Predicciones Futuras'))
    fig.update_layout(title=f'Predicciones Futuras usando {model_name}', xaxis_title='Fecha', yaxis_title='Precio Predicho')
    
    return fig

# Callback para predicción en una fecha específica
@app.callback(
    Output('prediction-output', 'children'),
    [Input('predict-button', 'n_clicks')],
    [State('model-selector', 'value'), State('date-input', 'value')]
)
def predict_specific_date(n_clicks, model_name, date_input):
    if n_clicks > 0 and date_input:
        try:
            specific_date = datetime.strptime(date_input, '%Y-%m-%d')
            days_diff = (specific_date - last_date).days
            if days_diff <= 0:
                return f"Error: La fecha ingresada {date_input} debe ser posterior a la última fecha registrada {last_date.date()}."
            
            model = models[model_name]
            last_values = data.iloc[-1].drop(['Date', 'Price Tomorrow']).values.reshape(1, -1)
            prediction = model.predict(last_values)[0]
            
            return f"Predicción para {date_input} usando {model_name}: {prediction:.2f}"
        except ValueError:
            return "Error: Formato de fecha incorrecto. Usa YYYY-MM-DD."
    
    return ""

# Ejecutar la aplicación
if __name__ == '__main__':
    app.run_server(debug=True)
