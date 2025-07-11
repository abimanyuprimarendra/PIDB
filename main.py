import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
from sklearn.metrics.pairwise import cosine_similarity

st.title("Sistem Rekomendasi Tempat Wisata Yogyakarta")

# Fungsi untuk load CSV dari Google Drive
def load_csv_from_drive(file_id):
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = requests.get(url)
    if response.status_code != 200:
        st.error("Gagal mengunduh file dari Google Drive.")
        return pd.DataFrame()
    return pd.read_csv(io.StringIO(response.content.decode('utf-8')))

# Fungsi Reverse Geocoding
def get_address_from_coordinates(lat, lon):
    try:
        url = f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}&format=json"
        headers = {'User-Agent': 'streamlit-app'}
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data.get('display_name', 'Alamat tidak ditemukan')
        else:
            return "Gagal mengambil alamat"
    except:
        return "Tidak bisa mengakses alamat"

# Ganti dengan file_id kamu
tour_csv_id = '11hQi3aqQkq5m2567jl7Ux1klXShLnYox'
rating_csv_id = '14Bke4--cJi6bVrQng8HlpFihOFOPhGZJ'

# Load dataset
tours_df = load_csv_from_drive(tour_csv_id)
ratings_df = load_csv_from_drive(rating_csv_id)

if tours_df.empty or ratings_df.empty:
    st.stop()

# Preprocessing
pivot_table = ratings_df.pivot_table(index='User_Id', columns='Place_Id', values='Place_Ratings').fillna(0)
item_similarity_matrix = cosine_similarity(pivot_table.T)
item_similarity_df = pd.DataFrame(item_similarity_matrix, index=pivot_table.columns, columns=pivot_table.columns)
place_names = tours_df.set_index('Place_Id')['Place_Name'].to_dict()

# Fungsi prediksi rating
def predict_rating(user_id, item_id, pivot_df, similarity_df):
    if item_id not in similarity_df.columns or user_id not in pivot_df.index:
        return np.nan
    user_ratings = pivot_df.loc[user_id]
    item_similarities = similarity_df[item_id]
    rated_items = user_ratings[user_ratings > 0].index
    if len(rated_items) == 0:
        return np.nan
    sim_scores = item_similarities[rated_items]
    ratings = user_ratings[rated_items]
    if sim_scores.sum() == 0:
        return np.nan
    return np.dot(sim_scores, ratings) / sim_scores.sum()

# Fungsi rekomendasi tempat wisata serupa
def recommend_similar_places_by_category(place_id, similarity_df, tours_df, top_n=5):
    if place_id not in similarity_df or place_id not in tours_df['Place_Id'].values:
        return pd.DataFrame()
    selected_category = tours_df.loc[tours_df['Place_Id'] == place_id, 'Category'].values[0]
    sim_scores = similarity_df[place_id].sort_values(ascending=False)
    candidate_ids = sim_scores.iloc[1:top_n*3].index
    candidate_df = tours_df[tours_df['Place_Id'].isin(candidate_ids)]
    filtered_df = candidate_df[candidate_df['Category'] == selected_category]
    return filtered_df[['Place_Id', 'Place_Name', 'Category', 'City']].head(top_n).reset_index(drop=True)

# Sidebar navigasi
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman", ["Prediksi Rating", "Rekomendasi Tempat Serupa"])

# Halaman Prediksi Rating
if page == "Prediksi Rating":
    st.header("Prediksi Rating Tempat Wisata")

    user_id = st.sidebar.selectbox("Pilih User_Id", pivot_table.index)
    place_id = st.sidebar.selectbox("Pilih Place_Id", pivot_table.columns, format_func=lambda x: f"{x} - {place_names.get(x, 'Unknown')}")

    pred = predict_rating(user_id, place_id, pivot_table, item_similarity_df)

    if np.isnan(pred):
        st.warning("Tidak dapat memprediksi rating untuk kombinasi tersebut.")
    else:
        st.success(f"Prediksi rating User {user_id} untuk tempat **{place_names.get(place_id)}** adalah **{pred:.2f}**")

# Halaman Rekomendasi Tempat Serupa
elif page == "Rekomendasi Tempat Serupa":
    st.header("Rekomendasi Tempat Wisata Serupa")

    place_name_to_id = {v: k for k, v in place_names.items()}
    selected_place_name = st.selectbox("Pilih Nama Tempat Wisata", list(place_name_to_id.keys()))
    selected_place_id = place_name_to_id[selected_place_name]

    st.info(f"Menampilkan rekomendasi serupa untuk: **{selected_place_name}**")

    recs = recommend_similar_places_by_category(selected_place_id, item_similarity_df, tours_df, top_n=5)

    if recs.empty:
        st.info("Tidak ditemukan rekomendasi.")
    else:
        for _, row in recs.iterrows():
            st.markdown("---")
            st.subheader(row['Place_Name'])
            st.markdown(f"**Kategori:** {row['Category']}  \n**Kota:** {row['City']}")

            # Deskripsi
            if 'Description' in tours_df.columns:
                deskripsi = tours_df.loc[tours_df['Place_Id'] == row['Place_Id'], 'Description'].values[0]
                st.markdown(f"**Deskripsi:** {deskripsi}")

            # Koordinat dan alamat
            if 'Latitude' in tours_df.columns and 'Longitude' in tours_df.columns:
                lat = tours_df.loc[tours_df['Place_Id'] == row['Place_Id'], 'Latitude'].values[0]
                lon = tours_df.loc[tours_df['Place_Id'] == row['Place_Id'], 'Longitude'].values[0]
            
                address = get_address_from_coordinates(lat, lon)
                st.markdown(f"**Alamat:** {address}")
