import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from joblib import dump

# Cargar los datos de entrenamiento
X_train = pd.read_csv('../../seleccion_caracteristicas_relacionadas/gold_price_features_train.csv')
y_train = pd.read_csv('../../seleccion_caracteristicas_relacionadas/gold_price_target_train.csv')

# Cargar los datos de prueba
X_test = pd.read_csv('../../seleccion_caracteristicas_relacionadas/gold_price_features_test.csv')
y_test = pd.read_csv('../../seleccion_caracteristicas_relacionadas/gold_price_target_test.csv')

# Entrenar el modelo de regresión lineal
model = LinearRegression()
model.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# Evaluar el modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# Guardar el modelo entrenado en un archivo
dump(model, 'gold_price_model.pkl')

print("Modelo entrenado y guardado exitosamente en 'gold_price_model.pkl'.")
