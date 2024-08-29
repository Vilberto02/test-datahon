import pandas as pd
import numpy as np
import csv

df = pd.read_csv("../../../data/WMT.csv")

# Valores iniciales del csv
# print(df.head())

# Valores estadísticos de los datos de cada columna.
# print(df.describe())

# Conteo de valores nulos en cada columna
# print(df.isnull().sum()) # -> output: 0 en cada columna. Los datos estan limpios.

# Copia del dataframe
df_processed = df.copy()

# Guardar la CSV
df_processed.to_csv("data_processed.csv", index=False, encoding='utf-8')