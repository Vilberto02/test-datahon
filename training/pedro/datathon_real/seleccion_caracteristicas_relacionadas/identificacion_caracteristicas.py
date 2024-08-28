import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar los datos limpios
cleaned_data_path = "../limpieza_datos/Gold_Price_Prediction_Cleaned.csv"
data = pd.read_csv(cleaned_data_path)

# Calcular la correlación entre las variables
correlation_matrix = data.corr()

# Visualizar la matriz de correlación
plt.figure(figsize=(14, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Matriz de Correlación de las Características')
plt.show()

# Seleccionar las características más relevantes basadas en la correlación
# Asumiendo que la columna 'Price Tomorrow' es el objetivo, buscamos las características más correlacionadas
target = 'Price Tomorrow'
correlation_with_target = correlation_matrix[target].sort_values(ascending=False)

# Mostrar las características más correlacionadas con el objetivo
print("Características más correlacionadas con el precio del oro:")
print(correlation_with_target)
