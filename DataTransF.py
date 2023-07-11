import pandas as pd
import numpy as np
import json
import ast
import re

#Lectura de datos
data= pd.read_csv('movies_dataset.csv', low_memory= False)
data2= pd.read_csv('credits.csv', low_memory= False)

#Coerción a dato numérico
data['id'] = pd.to_numeric(data['id'], errors='coerce')
#Uniendo data
dataF= pd.merge(data, data2, on='id', how='inner')
#Eliminando columnas que no son necesarias
colSub = ['video', 'imdb_id', 'adult', 'original_title', 'poster_path', 'homepage']
dataF = dataF.drop(colSub, axis=1)

#Clase para desencriptar data 
class TransF:
    
    @staticmethod
    def ConvertStr(value):
        '''Verifica si el valor es una lista o
         un diccionario y lo convierte en una representación de 
         cadena de JSON utilizando json.dumps()'''
        if isinstance(value, (list, dict)):
            return json.dumps(value)
        return str(value)
    
    
    @staticmethod
    def ExtractName(value):
        '''Función que busca todas
         las coincidencias de texto que sigan un patrón específico 
         utilizando expresiones regulares. 
         '''
        pattern = r"'name': '([^']*)'"
        CondC = re.findall(pattern, value)
        if len(CondC) > 0:
            name = CondC[0]
            return name
        else:
            return None
    
    @staticmethod
    def ExtractDirect(value):
        '''Funcion que define un patrón de expresión regular, 
        coincidencias de texto que sigan el formato 'name': 'value', 
        extrayendo el 'name' '''
        pattern = r"'Director', 'name': '([^']*)'"
        CondC = re.findall(pattern, value)
        if len(CondC) > 0:
            name = CondC[0]
            return name
        else:
            return None
    
    
    @staticmethod
    def ConvToDicc(column):
        '''Aplica una transformación a cada valor de una
        columna utilizando el método apply.'''
        return column.apply(lambda x: ast.literal_eval(x) if pd.notna(x) else np.nan)
    
    
    @staticmethod
    def DesCol(column):
        '''Desanida una columna que contiene listas de diccionarios.'''
        return column.apply(lambda x: ', '.join([d['name'] for d in x]) if isinstance(x, list) else np.nan)
    
    
    @staticmethod
    def ConvertToDiccBtc(column):
        '''Convierte los valores de una columna en objetos de diccionario.'''
        return column.apply(lambda x: {} if pd.isna(x) else ast.literal_eval(x))
    
    @staticmethod
    def DesBtc(column):
      return column.apply(lambda x: x['name'] if isinstance(x, dict) and 'name' in x else np.nan)

#Desanidando datos

dataF['belongs_to_collection'] = TransF.ConvertToDiccBtc(dataF['belongs_to_collection'])
dataF['belongs_to_collection'] = TransF.DesBtc(dataF['belongs_to_collection'])

dataF['production_companies'] = TransF.ConvToDicc(dataF['production_companies'])
dataF['production_companies'] = TransF.DesCol(dataF['production_companies'])

dataF['production_countries'] = TransF.ConvToDicc(dataF['production_countries'])
dataF['production_countries'] = TransF.DesCol(dataF['production_countries'])

dataF['spoken_languages'] = dataF['spoken_languages'].apply(TransF.ConvertStr).apply(TransF.ExtractName)

dataF['director'] = dataF['crew'].apply(TransF.ConvertStr).apply(TransF.ExtractDirect)

dataF = dataF.drop('crew', axis=1)

dataF['cast'] = TransF.ConvToDicc(dataF['cast'])
dataF['cast'] = TransF.DesCol(dataF['cast'])

dataF['genres'] = TransF.ConvToDicc(dataF['genres'])
dataF['genres'] = TransF.DesCol(dataF['genres'])

#Llenando campos nulos con 0
dataF['revenue']= dataF['revenue'].fillna(0)
dataF['budget']= dataF['budget'].fillna(0)

#Borrando valores nulos
dataF= dataF.dropna(subset=['release_date'])

#Convirtiendo a fechas
dataF['release_date'] = pd.to_datetime(dataF['release_date'])
dataF['release_year']= dataF['release_date'].dt.year

#Creando campo 'return'
dataF['return']= dataF['revenue'] / dataF['budget'].astype('float')


dataF = dataF.reset_index()

partes = np.array_split(dataF, 2)  # Dividir en 2 partes pues en github no se sube el archivo completo

for i, parte in enumerate(partes):
    nombre_archivo = f"DataFilter{i+1}.csv"
    parte.to_csv(nombre_archivo, index=False)






