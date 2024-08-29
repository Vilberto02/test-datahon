# Website: Visualización de Datos de Walmart con Dash y Plotly

Este pagina web utiliza Dash, un framework de Python basado en Flask, para representar de manera interactiva los datos de un dataset de Walmart. La visualización de datos es realizada con gráficos de Plotly, lo que permite a los usuarios explorar y analizar los datos que se encuentran en el dataset.

## Descripción de la pagina web

El objetivo de esta aplicación web es proporcionar una interfaz gráfica interactiva que facilite el análisis de los datos de ventas de Walmart. La página web incluye varios gráficos interactivos que permiten a los usuarios visualizar métricas clave, como precios, ventas, volumen y tendencias a lo largo del tiempo.

## Características Principales

- **Gráficos Interactivos**: Utiliza Plotly para crear gráficos interactivos que permiten hacer zoom, filtrar datos, y visualizar información específica en detalle.
- **Interfaz Amigable**: Construida con Dash, la interfaz es fácil de usar y se adapta a las necesidades del usuario.
- **Filtros Personalizados**: Permite a los usuarios seleccionar diferentes categorías, productos, y fechas para personalizar la visualización.
- **Actualización en Tiempo Real**: Los gráficos se actualizan automáticamente al cambiar los filtros, proporcionando una experiencia dinámica de análisis de datos.

## Instalación

Para ejecutar esta aplicación, necesitas tener instaladas las bibliotecas necesarias. Puedes instalarlas utilizando `pip`:

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

### 6. Abre tu navegador y navega al enlace generado para visualizar la página web.
