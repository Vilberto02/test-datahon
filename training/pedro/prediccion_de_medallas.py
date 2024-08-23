import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle

# 1. Cargar los datos
data = pd.read_csv('ruta/al/archivo/Olympics_2024.csv')

# 2. Exploración de los datos
print(data.info())
print(data.describe())
print(data.head())

# 3. Limpieza de los datos
# - Eliminar duplicados, si los hay
data = data.drop_duplicates()

# - Manejo de valores faltantes (en este caso, no se encontraron valores nulos, pero se puede aplicar la siguiente línea si es necesario)
# data = data.dropna()

# - Corrección de tipos de datos (si es necesario)
# Ejemplo: data['Rank'] = data['Rank'].astype(int)

# 4. Preparación de los datos
# Selección de características y variable objetivo
X = data[['Gold', 'Silver', 'Bronze']]
y = data['Total']

# División en conjuntos de entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Entrenamiento del modelo
model = LinearRegression()
model.fit(X_train, y_train)

# 6. Evaluación del modelo
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print(f'MAE: {mae}')
print(f'RMSE: {rmse}')
print(f'R2 Score: {r2}')

# 7. Guardar el modelo entrenado
with open(b'C:\Users\anycodef\Documents\github\PedroSota\test-datathon\training\pedro\modelo_entrenado\olympics_medal_prediction_model.pkl', 'wb') as file:
    pickle.dump(model, file)

# Guardar los datos de entrenamiento para su uso posterior
X_train.to_csv('ruta/al/archivo/olympics_training_data.csv', index=False)
