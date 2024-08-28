import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from joblib import dump

# Cargar los datos limpios generados en el preprocesamiento
data = pd.read_csv('gold_price_cleaned.csv')

# Convertir la columna de fecha a formato ordinal (número de días desde una fecha base)
data['Date_Ordinal'] = pd.to_datetime(data['Date']).map(lambda date: date.toordinal())

# Seleccionar características y la variable objetivo
features = ['Date_Ordinal', 'Price 2 Days Prior', 'Price 1 Day Prior', 'Price Today', 
            'Twenty Moving Average', 'Fifty Day Moving Average', 'EFFR Rate', 
            'Volume', 'Crude', 'DXY']
X = data[features]
y = data['Price Tomorrow']

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo de árbol de decisión
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# Evaluar el modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# Guardar el modelo entrenado en un archivo
dump(model, 'gold_price_decision_tree_model.pkl')

print("Modelo de Árbol de Decisión entrenado y guardado exitosamente en 'gold_price_decision_tree_model.pkl'.")
