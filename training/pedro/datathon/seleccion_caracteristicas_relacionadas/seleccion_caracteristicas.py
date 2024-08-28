import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar los datos limpios
file_path = "../limpieza_datos/Gold_Price_Prediction_Cleaned.csv"
data = pd.read_csv(file_path)

# Calcular la matriz de correlación
correlation_matrix = data.corr()

# Selección de las características con alta correlación con el precio del oro "Price Tomorrow"
target = 'Price Tomorrow'
threshold = 0.5  # Umbral de correlación absoluta
relevant_features = correlation_matrix[target][abs(correlation_matrix[target]) > threshold].index.tolist()

# Filtrar el DataFrame con solo las características relevantes
selected_data = data[relevant_features]

# Visualizar las características seleccionadas
print(f"Características seleccionadas: {relevant_features}")

# Guardar las características seleccionadas en un nuevo archivo CSV
output_path = "Selected_Features_Gold_Price.csv"
selected_data.to_csv(output_path, index=False)
print(f"Los datos con las características seleccionadas se han guardado en: {output_path}")
