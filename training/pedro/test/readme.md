# Predicción de Medallas en los Juegos Olímpicos

## Descripción del Proyecto

Este proyecto se enfoca en predecir el número total de medallas que un país ganará en los Juegos Olímpicos de 2024. Utilizando datos históricos de medallas ganadas por cada país en diferentes competiciones, se ha entrenado un modelo de regresión lineal para realizar predicciones precisas.

El script principal, `prediccion_de_medallas.py`, carga los datos, realiza la limpieza y preparación necesaria, entrena un modelo de machine learning, y finalmente evalúa y guarda el modelo entrenado.

## Estructura del Proyecto

El proyecto está organizado de la siguiente manera:

- `prediccion_de_medallas.py`: Script principal para la limpieza de datos, entrenamiento del modelo y predicción.
- `requirements.txt`: Archivo que contiene las dependencias necesarias para ejecutar el script.
- `olympics_training_data.csv`: Archivo CSV que contiene los datos de entrenamiento.
- `olympics_medal_prediction_model.pkl`: Archivo que contiene el modelo de regresión lineal entrenado.

## Dependencias

Para ejecutar este proyecto, es necesario instalar las siguientes dependencias:

- `pandas`: Para manipulación y análisis de datos.
- `scikit-learn`: Para entrenamiento y evaluación de modelos de machine learning.
- `matplotlib`: Para visualización de datos.

Puedes instalar todas las dependencias ejecutando el siguiente comando:

```bash
pip install -r requirements.txt
