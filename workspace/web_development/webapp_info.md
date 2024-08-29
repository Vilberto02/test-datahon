# Website: Visualización y Predicción del Precio del Oro con Dash y Plotly

Esta página web usa Dash, un framework de Python, para representar los datos de un dataset sobre el precio del oro mediante gráficos interactivos de Plotly. Además, se han integrado modelos de predicción ya entrenados y gráficos estadísticos como la matriz de correlación y el gráfico de dispersión para un análisis más profundo.

## Descripción del Proyecto

La finalidad de esta aplicación web es proporcionar una herramienta interactiva que permita analizar y predecir los precios del oro en base al dataset proporcionado.

## Características Principales

- **Gráficos Interactivos**: Implementación de gráficos de línea, de dispersión y otros, utilizando Plotly para permitir una visualización detallada de los datos históricos del precio del oro.
- **Modelos de Predicción**: Incorporación de modelos de predicción previamente entrenados para estimar los precios futuros del oro, lo que facilita la toma de decisiones basada en datos.
- **Análisis Estadístico**: Visualización de la matriz de correlación para identificar la relación entre diferentes variables y gráficos de dispersión para un análisis detallado de las tendencias.

## Instalación

Para ejecutar esta aplicación, asegúrate de tener instaladas las siguientes bibliotecas. Puedes instalarlas utilizando `pip`:

### 1. Instalar una librería para crear entornos virtuales

```bash
pip install virtualenv
```

### 2. Crear un entorno virtual

```bash
virtualenv venv
```

### 3. Activar el entorno virtual

```bash
source venv/Scripts/activate
```

**Nota:** _De aquí en adelante se usa el entorno virtual para la ejecución de la aplicación web_

### 4. Instalar las dependencias

```bash
pip install -r requirements.txt
```

### 5. Ejecutar la aplicación web

```bash
python.exe app.py
```

### 6. Abre tu navegador y navega al enlace que se genero para ver la página web.
