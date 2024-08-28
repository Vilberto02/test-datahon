import pandas as pd

# Cargar el dataset
data = pd.read_csv('../gold_price_prediction.csv')

# Preprocesamiento de datos
# Convertir la columna de fecha a formato datetime y luego a un formato numérico para usar como característica
data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%y')
data['Date_Ordinal'] = data['Date'].map(lambda date: date.toordinal())

# Eliminar filas con valores nulos, especialmente en la columna 'Price Tomorrow'
data = data.dropna(subset=['Price Tomorrow'])

# Guardar los datos limpios en un nuevo archivo CSV
data.to_csv('gold_price_cleaned.csv', index=False)

print("Datos limpios guardados exitosamente en 'gold_price_cleaned.csv'.")
