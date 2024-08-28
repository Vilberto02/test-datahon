from flask import Flask, render_template
from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor


server = Flask(__name__)

app = Dash(__name__, server=server, url_base_pathname='/dash/')

df = pd.read_csv('../data/Gold_Price_Prediction.csv')
df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")


# def train_model():

#   return model

app.layout = html.Div([
  html.Div([
    html.H1(children='Visualización de los datos', style={'textAlign': 'center'}),
  ])
])


@server.route('/')
def home():
    return render_template('index.html')


@server.route('/dash/')
def dash_app():
    return app.index()

if __name__ == '__main__':
    server.run(debug=True)