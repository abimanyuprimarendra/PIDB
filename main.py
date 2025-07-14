import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Rekomendasi Wisata Jogja", layout="centered")
st.title("🎯 Sistem Rekomendasi Tempat Wisata di Yogyakarta")
st.caption("Metode: Item-Based Collaborative Filtering (IBCF)")

# =========================
# 1. Load Data dari Google Drive
# =========================
def load_csv_from_drive(file_id):
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = requests.get(url)
    if response.status_code != 200:
        st.error("❌ Gagal mengunduh file dari Google Drive.")
        return pd.DataFrame()
    return pd.read_csv(io.StringIO(response.content.decode('utf-8')))

# File ID Google Drive
tour_csv_id = '11hQi3aqQkq5m2567jl7Ux1klXShLnYox'
rating_csv_id = '14Bke4--cJi6bVrQng8HlpFihOFOPhGZJ'

# Load datasets
tour_df = load_csv_from_drive(tour_csv_id)
rating_df = load_csv_from_drive(rating_csv_id)

# =========================
# 2. Preprocessing
# =========================
tour_df.dropna(subset=['Place_Id', 'Place_Name'], inplace=True)
rating_df.dropna(subset=['User_Id', 'Place_Id', 'Place_Ratings'], inplace=True)

tour_df['Place_Id'] = tour_df['Place_Id'].astype(int).astype(str)
rating_df['Place_Id'] = rating_df['Place_Id'].astype(int).astype(str)
rating_df['User_Id'] = rating_df['User_Id'].astype(str)

# =========================
# 3. Matriks Rating & Similarity
# =========================
rating_matrix = rating_df.pivot_table(index='Place_Id', columns='User_Id', values='Place_Ratings').fillna(0)
item_similarity = cosine_similarity(rating_matrix)
item_similarity_df = pd.DataFrame(item_similarity, index=rating_matrix.index, columns=rating_matrix.index)

# =========================
# 4. Fungsi Rekomendasi
# =========================
def get_recommendations(place_id, top_n=5):
    place_id = str(int(place_id))
    if place_id not in item_similarity_df.index:
        return pd.DataFrame()

    similar_scores = item_similarity_df[place_id].sort_values(ascending=False).drop(place_id)
    top_places = similar_scores.head(top_n).index.tolist()

    return tour_df[tour_df['Place_Id'].isin(top_places)][
        ['Place_Name', 'Category', 'City', 'Rating']
    ]

def get_recommendation_by_name(place_name, top_n=5):
    match = tour_df[tour_df['Place_Name'].str.lower() == place_name.lower()]
    if match.empty:
        st.warning(f"Tempat '{place_name}' tidak ditemukan.")
        return pd.DataFrame()
    
    place_id = match['Place_Id'].values[0]
    origin = match.iloc[0]
    st.success(f"📍 Tempat Asal: {origin['Place_Name']} ({origin['Category']}, {origin['City']})")
    return get_recommendations(place_id, top_n)

# =========================
# 5. UI Streamlit
# =========================
st.subheader("Cari Rekomendasi Berdasarkan Nama Tempat")

place_names = sorted(tour_df['Place_Name'].unique())
selected_place = st.selectbox("Pilih Tempat Wisata", place_names)

if st.button("Tampilkan Rekomendasi"):
    rekomendasi_df = get_recommendation_by_name(selected_place)
    if not rekomendasi_df.empty:
        st.write("🎯 **5 Tempat Wisata yang Direkomendasikan:**")
        st.dataframe(rekomendasi_df, use_container_width=True)
