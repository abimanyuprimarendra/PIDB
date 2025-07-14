You said:
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

# Sidebar pencarian wisata
st.sidebar.title("Cari Tempat Wisata")
place_name_to_id = {v: k for k, v in place_names.items()}
selected_place_name = st.sidebar.selectbox("Pilih Tempat Wisata", list(place_name_to_id.keys()))
selected_place_id = place_name_to_id[selected_place_name]

# Tampilan detail tempat yang dipilih
st.header(f"Tempat Wisata: {selected_place_name}")
selected_row = tours_df[tours_df['Place_Id'] == selected_place_id].iloc[0]

st.markdown(f"**Kategori:** {selected_row['Category']}  \n**Kota:** {selected_row['City']}")
if 'Description' in selected_row:
    st.markdown(f"**Deskripsi:** {selected_row['Description']}")
if 'Latitude' in selected_row and 'Longitude' in selected_row:
    lat, lon = selected_row['Latitude'], selected_row['Longitude']
    st.markdown(f"**Koordinat:** {lat}, {lon}")
    address = get_address_from_coordinates(lat, lon)
    st.markdown(f"**Alamat:** {address}")

# Rekomendasi
st.subheader("Rekomendasi Tempat Serupa")
recs = recommend_similar_places_by_category(selected_place_id, item_similarity_df, tours_df, top_n=5)

if recs.empty:
    st.info("Tidak ditemukan rekomendasi.")
else:
    for _, row in recs.iterrows():
        st.markdown("---")
        st.subheader(row['Place_Name'])
        st.markdown(f"**Kategori:** {row['Category']}  \n**Kota:** {row['City']}")
        if 'Description' in tours_df.columns:
            deskripsi = tours_df.loc[tours_df['Place_Id'] == row['Place_Id'], 'Description'].values[0]
            st.markdown(f"**Deskripsi:** {deskripsi}")
        if 'Latitude' in tours_df.columns and 'Longitude' in tours_df.columns:
            lat = tours_df.loc[tours_df['Place_Id'] == row['Place_Id'], 'Latitude'].values[0]
            lon = tours_df.loc[tours_df['Place_Id'] == row['Place_Id'], 'Longitude'].values[0]
            st.markdown(f"**Koordinat:** {lat}, {lon}")
            address = get_address_from_coordinates(lat, lon)
            st.markdown(f"**Alamat:** {address}")
