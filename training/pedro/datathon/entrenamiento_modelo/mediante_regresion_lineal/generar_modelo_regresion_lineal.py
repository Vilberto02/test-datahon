import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Cargar los datos con las características seleccionadas
file_path = "../../seleccion_caracteristicas_relacionadas/Selected_Features_Gold_Price.csv"
data = pd.read_csv(file_path)

# Definir las características (X) y el objetivo (y)
X = data.drop(columns=['Price Tomorrow'])
y = data['Price Tomorrow']

# Dividir los datos en conjuntos de entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicializar y entrenar el modelo de Regresión Lineal
model = LinearRegression()
model.fit(X_train, y_train)

# Realizar predicciones sobre el conjunto de prueba
y_pred = model.predict(X_test)

# Evaluar el modelo
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared (R²): {r2}")

# Guardar el modelo entrenado si se desea utilizar más adelante
import joblib
model_output_path = "Gold_Price_Prediction_Model.pkl"
joblib.dump(model, model_output_path)
print(f"Modelo entrenado guardado en: {model_output_path}")
