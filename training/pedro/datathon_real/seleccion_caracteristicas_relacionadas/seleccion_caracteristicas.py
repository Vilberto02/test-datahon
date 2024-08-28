import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Cargar los datos
file_path = "../datos_limpios/gold_price_prediction.csv"
data = pd.read_csv(file_path)

# Convertir la columna de fechas a formato datetime especificando el formato
if 'Date' in data.columns:
    # Ajusta el formato según la estructura de tus fechas, por ejemplo: '%m/%d/%y' para MM/DD/YY
    data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%y', errors='coerce')

# Imputar o eliminar valores nulos
numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = data.select_dtypes(include=['object']).columns

# Imputar valores nulos en columnas numéricas con la mediana
data[numeric_cols] = data[numeric_cols].apply(lambda col: col.fillna(col.median()))

# Convertir cualquier dato categórico que no sea de fecha a su tipo adecuado o imputar valores faltantes
data[categorical_cols] = data[categorical_cols].apply(lambda col: col.fillna(col.mode()[0]) if col.name != 'Date' else col)

# Tratamiento de outliers: eliminar filas con valores más allá de 3 desviaciones estándar
for col in numeric_cols:
    mean = data[col].mean()
    std = data[col].std()
    data = data[(data[col] >= mean - 3 * std) & (data[col] <= mean + 3 * std)]

# Seleccionar las características y el objetivo
relevant_features = [col for col in numeric_cols if col != 'Price Tomorrow'] + ['Price Tomorrow']
selected_data = data[relevant_features]

# Escalar las características
scaler = StandardScaler()
X = selected_data.drop(columns=['Price Tomorrow'])
y = selected_data['Price Tomorrow']

X_scaled = scaler.fit_transform(X)

# División en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Guardar los datos tratados
selected_data.to_csv("Treated_Gold_Price_Data.csv", index=False)
print("Los datos tratados se han guardado en: Treated_Gold_Price_Data.csv")

# Guardar los conjuntos de entrenamiento y prueba
train_data = pd.DataFrame(X_train, columns=X.columns)
train_data['Price Tomorrow'] = y_train.values
train_data.to_csv("Train_Gold_Price_Data.csv", index=False)

test_data = pd.DataFrame(X_test, columns=X.columns)
test_data['Price Tomorrow'] = y_test.values
test_data.to_csv("Test_Gold_Price_Data.csv", index=False)

print("Los conjuntos de entrenamiento y prueba se han guardado en:")
print("Train_Gold_Price_Data.csv y Test_Gold_Price_Data.csv")
