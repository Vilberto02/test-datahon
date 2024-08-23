import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle

# 1. Cargar y limpiar los datos
def cargar_datos(ruta_csv):
    data = pd.read_csv(ruta_csv)
    data = data.drop_duplicates()
    return data

# 2. Entrenar el modelo
def entrenar_modelo(data):
    X = data[['Gold', 'Silver', 'Bronze']]
    y = data['Total']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Evaluar el modelo
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)

    print(f'MAE: {mae}')
    print(f'RMSE: {rmse}')
    print(f'R2 Score: {r2}')
    
    return model

# 3. Guardar el modelo
def guardar_modelo(modelo, ruta_modelo):
    with open(ruta_modelo, 'wb') as file:
        pickle.dump(modelo, file)

# Main
if __name__ == "__main__":
    ruta_csv = './datos_de_entrenamiento/Olympics 2024.csv'
    ruta_modelo = './modelo_entrenado/olympics_medal_prediction_model.pkl'
    
    # Cargar y entrenar el modelo
    data = cargar_datos(ruta_csv)
    modelo = entrenar_modelo(data)
    
    # Guardar el modelo entrenado
    guardar_modelo(modelo, ruta_modelo)
    print(f"Modelo entrenado y guardado en {ruta_modelo}")
