import streamlit as st
import pandas as pd
import numpy as np
import time
from sklearn.metrics.pairwise import cosine_similarity
from heapq import nlargest

# ---------- LOAD DATASET ----------
@st.cache_data(show_spinner=True)
def load_data():
    csv_url_tour = "https://drive.google.com/uc?id=1toXFdx4bIbDevyPSmEdbs2gG3PR9iYI-"
    csv_url_rating = "https://drive.google.com/uc?id=1NUbzdY_ZNVI2Gc9avZaTvQNT6gp5tc4y"

    tour_df = pd.read_csv(csv_url_tour, dtype=str)
    rating_df = pd.read_csv(csv_url_rating, dtype=str)

    # Bersihkan spasi di kolom object
    tour_df = tour_df.apply(lambda col: col.str.strip() if col.dtype == 'object' else col)
    rating_df = rating_df.apply(lambda col: col.str.strip() if col.dtype == 'object' else col)

    # Konversi Place_Id dan User_Id: 
    # 1. jadi float dulu untuk hilangkan kemungkinan desimal '.0'
    # 2. drop rows yang tidak bisa konversi (NaN)
    # 3. convert ke int, lalu ke str supaya konsisten
    for df, cols in [(rating_df, ['User_Id', 'Place_Id']), (tour_df, ['Place_Id'])]:
        for col in cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')  # ke float, NaN jika gagal
        df.dropna(subset=cols, inplace=True)  # drop row yang ada NaN di kolom penting
        for col in cols:
            df[col] = df[col].astype(int).astype(str)

    # Pastikan Place_Ratings numeric dan bersih
    rating_df['Place_Ratings'] = pd.to_numeric(rating_df['Place_Ratings'], errors='coerce')
    rating_df.dropna(subset=['Place_Ratings'], inplace=True)
    rating_df.drop_duplicates(inplace=True)

    # Pastikan kolom lain di tour_df valid
    tour_df.dropna(subset=['Place_Name'], inplace=True)
    tour_df['Latitude'] = pd.to_numeric(tour_df['Latitude'], errors='coerce')
    tour_df['Longitude'] = pd.to_numeric(tour_df['Longitude'], errors='coerce')

    return tour_df, rating_df

# ---------- PREPARASI DATA & METODE PREDIKSI ----------
@st.cache_data(show_spinner=False)
def prepare_data(tour_df, rating_df):
    user_item_matrix = rating_df.pivot_table(index='User_Id', columns='Place_Id', values='Place_Ratings')
    user_item_matrix = user_item_matrix.fillna(0)
    item_similarity = user_item_matrix.corr(method='pearson')
    item_similarity = item_similarity.fillna(0)
    mean_ratings_dict = user_item_matrix.replace(0, np.nan).mean().to_dict()
    
    return user_item_matrix, item_similarity, mean_ratings_dict, tour_df, rating_df

# (Fungsi lain seperti precompute_top_k_neighbors, predict_rating_fast, evaluate_model, recommend_places tetap sama)

# ========== STREAMLIT UI ==========

st.title("Sistem Rekomendasi Tempat Wisata Yogyakarta")

with st.spinner('Loading data...'):
    tour_df, rating_df = load_data()
    user_item_matrix, item_similarity_filled, mean_ratings_dict, tour_df, rating_df = prepare_data(tour_df, rating_df)

k = st.sidebar.slider("Pilih nilai k (jumlah tetangga):", 2, 10, 6)
top_k_neighbors = precompute_top_k_neighbors(item_similarity_filled, k=k)

@st.cache_data
def cached_evaluate():
    return evaluate_model(user_item_matrix, rating_df, mean_ratings_dict, top_k_neighbors)

mae, rmse, waktu = cached_evaluate()

st.header("Rekomendasi Tempat Wisata")

place_options = tour_df[['Place_Id', 'Place_Name']].drop_duplicates().sort_values('Place_Name')
place_name_list = place_options['Place_Name'].tolist()
place_id_list = place_options['Place_Id'].tolist()

selected_place_name = st.selectbox("Pilih Tempat Wisata:", place_name_list)
selected_place_id = place_options[place_options['Place_Name'] == selected_place_name]['Place_Id'].values[0]

recommended_places = recommend_places(selected_place_id, top_k_neighbors, tour_df, k=5)

st.subheader(f"Rekomendasi tempat mirip dengan {selected_place_name}:")
for idx, row in recommended_places.iterrows():
    st.markdown(f"**{row['Place_Name']}**")

st.markdown("---")
st.subheader("Evaluasi Model")
st.write(f"MAE: {mae:.4f}")
st.write(f"RMSE: {rmse:.4f}")
st.write(f"Waktu Prediksi: {waktu:.2f} ms")

map_df = recommended_places[['Latitude', 'Longitude', 'Place_Name']].copy()
selected_place_coords = tour_df[tour_df['Place_Id'] == selected_place_id][['Latitude', 'Longitude']]

if not selected_place_coords.empty:
    selected_place_coords = selected_place_coords.iloc[0]
    map_df = pd.concat([
        map_df,
        pd.DataFrame({
            'Latitude': [selected_place_coords['Latitude']],
            'Longitude': [selected_place_coords['Longitude']],
            'Place_Name': [selected_place_name]
        })
    ])

map_df = map_df.dropna(subset=['Latitude', 'Longitude'])

if not map_df.empty:
    st.subheader("Peta Lokasi Tempat Wisata")
    st.map(map_df.rename(columns={'Latitude': 'lat', 'Longitude': 'lon'}))
else:
    st.write("Data koordinat lokasi tidak tersedia untuk peta.")
