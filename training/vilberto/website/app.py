from flask import Flask, render_template
from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor

server = Flask(__name__)

app = Dash(__name__, server=server, url_base_pathname='/dash/')

df = pd.read_csv('../code/data_processed.csv')
df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")

def train_model():
    X = df[['Open', 'High', 'Low', 'Volume']]
    y = df['Close']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    # Algoritmo de Gradient Boosting: Usa modelos predictivos débiles (árboles de decisión) para que luego se creen modelo fuertes.
    model = HistGradientBoostingRegressor()
    model.fit(X_train, y_train)

    return model

model = train_model()


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
                className="date-picker-range",
            ),
        ], style={'display': 'flex', 'gap':'0.5rem', 'align-items':'center','float': 'right'}),

    ], className="container-inputs"),

    html.Div([
        dcc.Graph(id='price-chart', style={'width':'100%', 'height': '400px'}),
        html.H3('Predicción del precio de cierre según la fecha', style={'textAlign': 'center'}),
        dcc.Graph(id='prediction-chart', style={'width':'100%', 'height': '400px'})
    ], style={'display': 'flex', 'flex-direction': 'column', 'gap': '1.2rem', 'margin': '0 0.8rem'}),

    html.A("Regresar", href="/", className="back-btn")
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
def update_price_graph(selected_price, start_date, end_date):
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

@app.callback(
    Output('prediction-chart', 'figure'),
    Input('date-picker-range', 'start_date'),
    Input('date-picker-range', 'end_date')
)
def update_prediction_graph(start_date, end_date):
    filtered_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    X_filtered = filtered_df[['Open', 'High', 'Low', 'Volume']]

    predictions = model.predict(X_filtered)

    pred_df = pd.DataFrame({
        'Date': filtered_df['Date'],
        'Predicted Close': predictions
    })

    formatted_start_date = pd.to_datetime(start_date).strftime('%Y-%m-%d')
    formatted_end_date = pd.to_datetime(end_date).strftime('%Y-%m-%d')

    fig = px.line(
        pred_df,
        x='Date',
        y='Predicted Close',
        title=f'Predicción: {formatted_start_date} hasta {formatted_end_date}'
    )

    fig.update_layout(xaxis_title='Fecha', yaxis_title='Predicción de Precio de Cierre', template='plotly_dark')

    fig.update_traces(line=dict(color='red'))

    return fig

@server.route('/')
def home():
    return render_template('index.html')

@server.route('/dash/')
def dash_app():
    return app.index()

if __name__ == '__main__':
    server.run(debug=True)
