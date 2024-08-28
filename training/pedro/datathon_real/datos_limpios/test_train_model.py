import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Cargar el dataset
data = pd.read_csv('gold_price_prediction.csv')

# Preprocesamiento de datos
# Convertir la columna de fecha a formato numérico para usar como característica
data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%y')
data['Date_Ordinal'] = data['Date'].map(lambda date: date.toordinal())

# Eliminar filas con valores nulos, especialmente en la columna 'Price Tomorrow'
data = data.dropna(subset=['Price Tomorrow'])

# Seleccionar características y la variable objetivo
features = ['Date_Ordinal', 'Price 2 Days Prior', 'Price 1 Day Prior', 'Price Today', 
            'Twenty Moving Average', 'Fifty Day Moving Average', 'EFFR Rate', 
            'Volume', 'Crude', 'DXY']
X = data[features]
y = data['Price Tomorrow']

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo de regresión lineal
model = LinearRegression()
model.fit(X_train, y_train)

# Realizar predicciones
y_pred = model.predict(X_test)

# Evaluar el modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# Visualización de resultados
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.xlabel('Actual Price Tomorrow')
plt.ylabel('Predicted Price Tomorrow')
plt.title('Actual vs Predicted Prices')
plt.show()


from joblib import dump

# Guarda el modelo entrenado en un archivo
dump(model, 'gold_price_model.pkl')

print("Modelo guardado exitosamente.")
