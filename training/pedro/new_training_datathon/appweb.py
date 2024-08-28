import dash
from dash import dcc, html, Input, Output
import pandas as pd
import datetime
from joblib import load
import plotly.graph_objs as go

# Cargar el modelo de árbol de decisión entrenado
model = load('gold_price_decision_tree_model.pkl')

# Función para convertir una fecha a su formato ordinal
def date_to_ordinal(date):
    return date.toordinal()

# Función para predecir el precio del oro dado una fecha
def predict_price(date):
    date_ordinal = date_to_ordinal(date)
    
    # Valores aproximados para las demás características (ajusta estos valores para más precisión)
    price_2_days_prior = 2400
    price_1_day_prior = 2380
    price_today = 2370
    twenty_moving_average = 2405
    fifty_day_moving_average = 2360
    effr_rate = 5.25
    volume = 90
    crude = 75
    dxy = 103.5

    # Crear el vector de características con la fecha ordinal y valores estimados
    features = [[date_ordinal, price_2_days_prior, price_1_day_prior, price_today, 
                 twenty_moving_average, fifty_day_moving_average, effr_rate, 
                 volume, crude, dxy]]

    # Realizar la predicción con el modelo de árbol de decisión
    predicted_price = model.predict(features)
    return predicted_price[0]

# Crear la aplicación Dash
app = dash.Dash(__name__)

# Layout de la aplicación
app.layout = html.Div([
    html.H1("Predicción del Precio del Oro"),
    dcc.DatePickerSingle(
        id='date-picker',
        date=datetime.date.today(),
        display_format='YYYY-MM-DD',
        placeholder='Seleccione una fecha'
    ),
    html.Div(id='prediction-output', style={'margin-top': '20px'}),
    dcc.Graph(id='prediction-graph', style={'margin-top': '20px'})
])

# Callback para actualizar la predicción y el gráfico
@app.callback(
    [Output('prediction-output', 'children'),
     Output('prediction-graph', 'figure')],
    [Input('date-picker', 'date')]
)
def update_prediction(selected_date):
    if selected_date:
        # Convertir la fecha seleccionada
        selected_date = datetime.datetime.strptime(selected_date, '%Y-%m-%d').date()
        predicted_price = predict_price(selected_date)

        # Generar predicciones para los próximos 15 días
        future_dates = [selected_date + datetime.timedelta(days=i) for i in range(15)]
        future_predictions = [predict_price(date) for date in future_dates]

        # Crear el gráfico de predicciones
        figure = {
            'data': [
                go.Scatter(
                    x=future_dates,
                    y=future_predictions,
                    mode='lines+markers',
                    name='Predicción de Precio'
                )
            ],
            'layout': go.Layout(
                title='Predicción del Precio del Oro para los Próximos 15 Días',
                xaxis={'title': 'Fecha'},
                yaxis={'title': 'Precio del Oro'}
            )
        }

        return f'El precio predicho del oro para la fecha {selected_date} es: {predicted_price:.2f}', figure
    else:
        return "Por favor, seleccione una fecha.", {}

# Ejecutar la aplicación
if __name__ == '__main__':
    app.run_server(debug=True)
