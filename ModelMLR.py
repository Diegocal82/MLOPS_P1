
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from PIL import Image
import streamlit as st

data1= pd.read_csv('DataFilter1.csv', encoding='latin-1', sep=',', low_memory=False)
data2= pd.read_csv('DataFilter2.csv', encoding='latin-1', sep=',', low_memory=False)
df= pd.concat([data1, data2], ignore_index=False)
#Extraer las columnas para ajustar el modelo
df = df[['title', 'belongs_to_collection', 'original_language', 'genres', 'overview', 'popularity', 'production_companies', 'production_countries', 'release_date', 'cast', 'director']]


# Combinar las características en un texto
df['features'] = df.apply(lambda x: ' '.join(x.values.astype(str)), axis=1)


# Crear la matriz TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['features'])


# Crear modelo de vecinos más cercanos
knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
# Entrenamos el modelo
knn_model.fit(tfidf_matrix)


def get_recommendations(title, knn_model, df, num_recommendations=5):
  # Obtener el índice de la película
    idx = df[df['title'] == title].index[0]

  # Encontrar los vecinos más cercanos
    distances, indices = knn_model.kneighbors(tfidf_matrix[idx], n_neighbors=num_recommendations+1)

    # Obtener los índices de las películas más similares
    movie_indices = indices.flatten()[1:]

    # Devolver las películas recomendadas
    return df['title'].iloc[movie_indices]

get_recommendations('Toy Story', knn_model, df)


#funcion recomendaciones para el usuario
def user_input():
    movie_title = st.text_input('Título de la película')

    if st.button('Obtener recomendaciones'):
        if movie_title:
            recommendations = get_recommendations(movie_title, knn_model, df)

            st.subheader(f'Recomendaciones para "{movie_title}":')
            st.write(recommendations)
        else:
            st.write('Ingrese el título de una película.')


# App principal
def main():
    #titulo de la aplicacion
    st.title('Movie Recommendation')
    user_input()

# Ejecutar la app principal
if __name__ == '__main__':
    main()