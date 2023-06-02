import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
from PIL import Image
from sklearn.neighbors import NearestNeighbors

st.set_page_config(
    page_title = "CreuseFlix : Vos recommnadations de films",
    page_icon = "🍿",
    layout="wide",
)

@st.cache_data
def load_data():
    rec_engine_df = pd.read_csv("rec_engine_streamlit2.csv")
    rec_engine_df['production_countries'] = rec_engine_df['production_countries'].fillna('')
    rec_engine_df = rec_engine_df[~rec_engine_df['production_countries'].str.contains('IN')]
    return rec_engine_df

df = load_data()

logo = Image.open('/Users/mickaelpouhaer/Documents/Python/C.png')
popcorn = Image.open('/Users/mickaelpouhaer/Documents/Python/popcorn_streamlit2.png')

N_RECOMMENDED_MOVIES = 6

FEATURES = ['weightedRating', 'Action', 'Adult', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime', 
          'Documentary', 'Drama', 'Family', 'Fantasy', 'Film-Noir', 'History', 'Horror', 'Music', 'Musical', 
          'Mystery', 'News', 'Reality-TV', 'Romance', 'Sci-Fi','Sport', 'Talk-Show', 'Thriller', 'War', 'Western']

def recommend_movies(movie_id_or_title):
    X = df[FEATURES]
    distanceKNN = NearestNeighbors(n_neighbors=N_RECOMMENDED_MOVIES + 1).fit(X)

    selected_movie = df[df['primaryTitle'] == movie_id_or_title]
    
    if not selected_movie.empty:
        X_selected = selected_movie[FEATURES]

        distances, indices = distanceKNN.kneighbors(X_selected)

        recommended_movies = df.iloc[indices[0]].reset_index(drop=True)

        # Exclude the selected movie from the recommendations
        recommended_movies = recommended_movies[recommended_movies['primaryTitle'] != movie_id_or_title]

        # Return a list of tuples, where each tuple is (movie title, average rating, poster path, overview)
        return list(zip(recommended_movies['primaryTitle'], 
                        recommended_movies['averageRating_y'], 
                        recommended_movies['poster_path'], 
                        recommended_movies['overview']))

    return []

BASE_URL = "https://image.tmdb.org/t/p/w600_and_h900_bestv2"

def random_movies(df, n=6):
    return df.sample(n)

def display_movies(recommendations):
    cols = st.columns(3)
    for i, (title, rating, poster_path, overview) in enumerate(recommendations):
        with cols[i % 3]:  # Use modulus to cycle through the columns
            st.markdown(f"<b>{title} (🍿 {rating})</b>", unsafe_allow_html=True)
            
            if poster_path:  # Only display the poster if the path is not None
                st.image(f"{BASE_URL}{poster_path}")
            with st.expander("Résumé"):
                st.write(overview)

with st.sidebar:
    st.image(logo)
    selected = option_menu (None, ['🍿 Recommandation','🍿 Selection aléatoire'], 
                            icons= ['🍿', '🍿'],
                            menu_icon="🍿")

if selected == '🍿 Recommandation':
    st.image(popcorn)
    movie_id_or_title = st.selectbox("Choisissez un titre de film :", df['primaryTitle'].unique())
    if movie_id_or_title:
        recommendations = recommend_movies(movie_id_or_title)
        display_movies(recommendations)

elif selected == '🍿 Selection aléatoire':
    st.image(popcorn)
    if st.button("Selection aléatoire de film"):
        random_selection = random_movies(df)
        recommendations = [(row['primaryTitle'], row['averageRating_y'], row['poster_path'], row['overview']) for _, row in random_selection.iterrows()]
        display_movies(recommendations)