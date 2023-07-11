# Proyecto Individual I
## Movie Recommendation 

Este es un proyecto de Machine Learning. El proyecto se desarrolló en varias etapas:

**Proceso de ETL (Extracción, Transformación, Carga):** En el archivo *DataTransF.py* se hicieron las operaciónes de depuración y limpieza de datos, los datos limpios se exportaron en formato *csv*, por efectos de capacidad de github se dividió el archivo en dos partes *DataFilter1.csv* y *DataFilter2.csv* con un peso menor a 25Mb.

**Análisis Exploratorio de Datos (EDA):** En el archivo *EDAP1.ipynb* se realizo un analisis exploratorio para encontrar posibles relaciónes entre las variables.

**Desarrollo del Modelo de Machine Learning:** En el archivo *ModelMLR.py* se implementó un modelo de recomendación usando el modelo de vecinos más cercanos (K-Nearest Neighbors), se implemento una interfaz grafica con la librería *streamlit* para visualizar en local.

**Implementación de la Interfaz:** En el archivo *main.py* se creó una interfaz con la librearía *FastAPI* que facilita la interacción de los usuarios con el modelo de machine learning y demas información util.

**Despliegue y Implementación:** Se realizo una implementación en un entorno de producción usando Render.

## Links

  * [API de Consultas](https://mlopsp1diego.onrender.com/)
    

