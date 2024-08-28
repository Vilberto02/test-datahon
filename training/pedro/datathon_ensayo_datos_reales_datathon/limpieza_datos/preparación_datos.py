import pandas as pd
from sklearn.preprocessing import StandardScaler

# Cargar los datos
file_path = "./Gold Price Prediction.csv"
data = pd.read_csv(file_path)

# Revisar si hay valores nulos y eliminarlos o imputarlos
print("Valores nulos antes de la limpieza:")
print(data.isnull().sum())

# Imputar valores nulos o eliminarlos (aquí simplemente eliminaremos filas con valores nulos)
data_cleaned = data.dropna()

# Convertir la columna de fechas a formato datetime
data_cleaned['Date'] = pd.to_datetime(data_cleaned['Date'], errors='coerce')

# Escalar los datos numéricos
scaler = StandardScaler()
scaled_features = scaler.fit_transform(data_cleaned.select_dtypes(include=['float64', 'int64']))

# Crear un DataFrame con los datos escalados
scaled_data = pd.DataFrame(scaled_features, columns=data_cleaned.select_dtypes(include=['float64', 'int64']).columns)

# Revisar los primeros registros del DataFrame limpio y escalado
print(scaled_data.head())


# Guardar los datos limpios y escalados en un nuevo archivo CSV
output_path = "Gold_Price_Prediction_Cleaned.csv"
scaled_data.to_csv(output_path, index=False)

print(f"Los datos limpios y escalados se han guardado en: {output_path}")
