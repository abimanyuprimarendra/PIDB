import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
from sklearn.metrics.pairwise import cosine_similarity

# Konfigurasi halaman
st.set_page_config(page_title="Rekomendasi Wisata Jogja", layout="wide")

# ===================================
# 1. Fungsi Load CSV dari Google Drive
# ===================================
def load_csv_from_drive(file_id):
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = requests.get(url)
    if response.status_code != 200:
        st.error("❌ Gagal mengunduh file dari Google Drive.")
        return pd.DataFrame()
    return pd.read_csv(io.StringIO(response.content.decode('utf-8')))

# ===================================
# 2. Load Data
# ===================================
tour_csv_id = '11hQi3aqQkq5m2567jl7Ux1klXShLnYox'
rating_csv_id = '14Bke4--cJi6bVrQng8HlpFihOFOPhGZJ'
tour_df = load_csv_from_drive(tour_csv_id)
rating_df = load_csv_from_drive(rating_csv_id)

# ===================================
# 3. Preprocessing
# ===================================
tour_df.dropna(subset=['Place_Id', 'Place_Name'], inplace=True)
rating_df.dropna(subset=['User_Id', 'Place_Id', 'Place_Ratings'], inplace=True)

tour_df['Place_Id'] = tour_df['Place_Id'].astype(int).astype(str)
rating_df['Place_Id'] = rating_df['Place_Id'].astype(int).astype(str)
rating_df['User_Id'] = rating_df['User_Id'].astype(str)

# ===================================
# 4. Matriks & Similarity
# ===================================
rating_matrix = rating_df.pivot_table(index='Place_Id', columns='User_Id', values='Place_Ratings').fillna(0)
item_similarity = cosine_similarity(rating_matrix)
item_similarity_df = pd.DataFrame(item_similarity, index=rating_matrix.index, columns=rating_matrix.index)

# ===================================
# 5. Fungsi Rekomendasi
# ===================================
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
        return pd.DataFrame(), None

    place_id = match['Place_Id'].values[0]
    origin = match.iloc[0]
    return get_recommendations(place_id, top_n), origin

# ===================================
# 6. Sidebar UI
# ===================================
st.sidebar.header("🎒 Pilih Tempat Wisata")
place_names = sorted(tour_df['Place_Name'].unique())
selected_place = st.sidebar.selectbox("Tempat Wisata", place_names)

# ===================================
# 7. Tampilkan Hasil di Halaman Utama
# ===================================
st.title("📍 Sistem Rekomendasi Tempat Wisata di Yogyakarta")

if selected_place:
    rekomendasi_df, origin_place = get_recommendation_by_name(selected_place)

    if origin_place is not None:
        st.markdown(f"### 🎯 5 Rekomendasi Mirip dengan: **{origin_place['Place_Name']}**")
        st.caption(f"Kategori: {origin_place['Category']} | Kota: {origin_place['City']}")

    if not rekomendasi_df.empty:
        st.markdown("### ✨ Rekomendasi Wisata Serupa")
        cols = st.columns(5)  # 5 kolom sejajar

        for idx, (_, row) in enumerate(rekomendasi_df.iterrows()):
            with cols[idx]:
                st.markdown(f"""
                <div style="background-color: #f9f9f9; border-radius: 15px; padding: 15px; 
                            box-shadow: 0 4px 8px rgba(0,0,0,0.1); min-height: 150px;">
                    <h4 style="margin-bottom: 10px;">{row['Place_Name']}</h4>
                    <p style="margin: 0;">Kategori: <b>{row['Category']}</b></p>
                    <p style="margin: 0;">Kota: <b>{row['City']}</b></p>
                    <p style="margin: 0;">⭐ Rating: <b>{row['Rating']}</b></p>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("Tidak ada rekomendasi ditemukan.")
