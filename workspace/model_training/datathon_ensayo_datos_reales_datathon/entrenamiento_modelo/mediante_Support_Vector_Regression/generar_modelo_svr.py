import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

# Cargar los datos con las características seleccionadas
file_path = "../../seleccion_caracteristicas_relacionadas/Selected_Features_Gold_Price.csv"
data = pd.read_csv(file_path)

# Definir las características (X) y el objetivo (y)
X = data.drop(columns=['Price Tomorrow'])
y = data['Price Tomorrow']

# Dividir los datos en conjuntos de entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar los datos (es esencial para SVR)
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()

# Inicializar y entrenar el modelo SVR
model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
model.fit(X_train_scaled, y_train_scaled)

# Realizar predicciones sobre el conjunto de prueba
y_pred_scaled = model.predict(X_test_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

# Evaluar el modelo
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared (R²): {r2}")

# Guardar el modelo entrenado si se desea utilizar más adelante
import joblib
model_output_path = "Gold_Price_SVR_Model.pkl"
joblib.dump(model, model_output_path)
print(f"Modelo entrenado guardado en: {model_output_path}")
