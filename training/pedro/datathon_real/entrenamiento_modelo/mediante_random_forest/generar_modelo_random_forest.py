import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Cargar los conjuntos de entrenamiento y prueba
train_data_path = "../../seleccion_caracteristicas_relacionadas/Train_Gold_Price_Data.csv"
test_data_path = "../../seleccion_caracteristicas_relacionadas/Test_Gold_Price_Data.csv"

train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)

# Separar las características (X) y el objetivo (y) en los conjuntos de entrenamiento y prueba
X_train = train_data.drop(columns=['Price Tomorrow'])
y_train = train_data['Price Tomorrow']
X_test = test_data.drop(columns=['Price Tomorrow'])
y_test = test_data['Price Tomorrow']

# Inicializar y entrenar el modelo de Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)
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
model_output_path = "Random_Forest_Model.pkl"
joblib.dump(model, model_output_path)
print(f"Modelo entrenado guardado en: {model_output_path}")
