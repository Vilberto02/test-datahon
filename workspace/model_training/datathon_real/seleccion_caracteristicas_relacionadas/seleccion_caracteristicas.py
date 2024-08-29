import pandas as pd
from sklearn.model_selection import train_test_split

# Cargar los datos limpios generados en el preprocesamiento
data = pd.read_csv('../datos_limpios/gold_price_cleaned.csv')

# Convertir la columna de fecha a formato numérico (ordinal) para usar como característica
data['Date_Ordinal'] = pd.to_datetime(data['Date']).map(lambda date: date.toordinal())

# Seleccionar características y la variable objetivo
features = ['Date', 'Price 2 Days Prior', 'Price 1 Day Prior', 'Price Today', 
            'Twenty Moving Average', 'Fifty Day Moving Average', 'EFFR Rate', 
            'Volume', 'Crude', 'DXY']

# Crear un nuevo DataFrame con las características seleccionadas y la variable objetivo
X = data[features]
y = data[['Price Tomorrow']]  # Columna objetivo

# Dividir los datos en conjuntos de entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Guardar los datos de entrenamiento y prueba en archivos CSV
X_train.to_csv('gold_price_features_train.csv', index=False)
X_test.to_csv('gold_price_features_test.csv', index=False)
y_train.to_csv('gold_price_target_train.csv', index=False)
y_test.to_csv('gold_price_target_test.csv', index=False)

print("Características y variable objetivo divididas en entrenamiento y prueba.")
print("Datos de entrenamiento guardados en 'gold_price_features_train.csv' y 'gold_price_target_train.csv'.")
print("Datos de prueba guardados en 'gold_price_features_test.csv' y 'gold_price_target_test.csv'.")
