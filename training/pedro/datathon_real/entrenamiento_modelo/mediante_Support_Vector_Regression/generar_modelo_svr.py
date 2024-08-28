import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
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

# Escalar los datos ya que SVR es sensible a la escala de las características
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()

# Inicializar y entrenar el modelo de SVR
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
model_output_path = "SVR_Model.pkl"
joblib.dump(model, model_output_path)
print(f"Modelo entrenado guardado en: {model_output_path}")
