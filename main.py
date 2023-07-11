from fastapi import FastAPI
import pandas as pd
import numpy as np
from typing import List
import uvicorn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


app = FastAPI()

data1= pd.read_csv('DataFilter1.csv', low_memory= False)
data2= pd.read_csv('DataFilter2.csv', low_memory= False)
data= pd.concat([data1, data2], ignore_index=False)

data['release_date'] = pd.to_datetime(data['release_date'])

#Extraer las columnas para ajustar el modelo
dataMl = data[['title', 'original_language', 'genres', 'overview', 'popularity', 'production_companies', 'production_countries', 'release_date', 'cast', 'director']]


# Combinar las características en un texto
dataMl['features'] = dataMl.apply(lambda x: ' '.join(x.values.astype(str)), axis=1)


# Crear la matriz TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(dataMl['features'])


# Crear modelo de vecinos más cercanos
knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
# Entrenar el modelo
knn_model.fit(tfidf_matrix)

#Crea todo el dataset en la api
@app.get("/dataframe", response_model=List[dict])
def get_dataframe():
    return data.to_dict(orient='records')


@app.get("/dataframe/month/{mes}", response_model=List)
def cantidad_filmaciones_mes(mes:str):
    def get_month(mes):
      meses = {
        'enero': 1,
        'febrero': 2,
        'marzo': 3,
        'abril': 4,
        'mayo': 5,
        'junio': 6,
        'julio': 7,
        'agosto': 8,
        'septiembre': 9,
        'octubre': 10,
        'noviembre': 11,
        'diciembre': 12
    }
      return meses[mes]

    N_movies= len(data[data['release_date'].dt.month == get_month(mes)]['id'].unique())

    return [f'{N_movies} películas fueron estrenadas en el mes de {mes}']

@app.get('/dataframe/day/{dia}', response_model= List)
def cantidad_filmaciones_dia(dia:str):
    def get_day(dia):
      DiaSemana = {
    'lunes': 'Monday',
    'martes': 'Tuesday',
    'miércoles': 'Wednesday',
    'jueves': 'Thursday',
    'viernes': 'Friday',
    'sábado': 'Saturday',
    'domingo': 'Sunday'
    }
      return DiaSemana[dia]

    N_movies= len(data[data['release_date'].dt.strftime('%A') == get_day(dia)]['id'].unique())

    return [f'{N_movies} películas fueron estrenadas en los días {dia}']

@app.get('/dataframe/score/{title}', response_model= List)
def score_titulo(title:str):
  year= data[data['title'] == title]['release_year'][0]
  score= data[data['title'] == title]['popularity'][0]
  return [f'La película {title} fue estrenada en el año {year} con un score/popularidad de {score}']

@app.get('/dataframe/votes/{title}', response_model= List)
def votos_titulo(title:str):
    TotV= data[data['title'] == title]['vote_count'][0]
    TotM= data[data['title'] == title]['vote_average'][0]
    year= data[data['title'] == title]['release_year'][0]
    if TotV >= 2000:
        return [f'La película {title} fue estrenada en el año {year}. La misma cuenta con un total de {TotV} valoraciones, con un promedio de {TotM}']
    else:
        return [f'La película {title} no cuenta con 2000 valoraciones']

@app.get('/dataframe/actor/{name}', response_model= List)
def get_actor(name:str):
  Filter = data['cast'].apply(lambda x: isinstance(x, str) and name in x.split(', '))
  dataF= data[Filter]
  NMov= len(dataF)
  Ret= dataF['return'][np.isfinite(dataF['return'])].sum()
  RetM= dataF['return'][np.isfinite(dataF['return'])].mean()

  return[f'El actor {name} ha participado de {NMov} cantidad de filmaciones, el mismo ha conseguido un retorno de {round(Ret, 3)} con un promedio de {round(RetM, 3)} por filmación']
  
@app.get("/dataframe/director/{director}", response_model= List)
def get_director(director:str):
    director_films = data[data['director'] == director]
    total_return = director_films['return'][np.isfinite(director_films['return'])].sum()
    presupuesto = director_films['budget'].sum()
    recaudacion = director_films['revenue'].sum()

    Restp = {'director': [director],'titulo': list(director_films['title']),'fecha': list(director_films['release_date']),'retorno': [round(total_return, 3)],'presupuesto': [presupuesto],'recaudación': [recaudacion]}
    return [f'{Restp}']

@app.get("/dataframe/Recomienda/{title}", response_model= List)
def recomendacion(title:str):
  # Obtener el índice de la película
  idx = dataMl[dataMl['title'] == title].index[0]

  # Encontrar los vecinos más cercanos
  distances, indices = knn_model.kneighbors(tfidf_matrix[idx], n_neighbors=6)

  # Obtener los índices de las películas más similares
  movie_indices = indices.flatten()[1:]

  # Devolver las películas recomendadas
  return [dataMl['title'].iloc[movie_indices].to_json(orient= 'records')]



uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
