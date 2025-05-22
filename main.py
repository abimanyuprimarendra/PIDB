import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load data
@st.cache_data(ttl=3600)
def load_data():
    csv_url_tour = "https://drive.google.com/uc?id=1toXFdx4bIbDevyPSmEdbs2gG3PR9iYI-"
    csv_url_rating = "https://drive.google.com/uc?id=1NUbzdY_ZNVI2Gc9avZaTvQNT6gp5tc4y"

    tour_df = pd.read_csv(csv_url_tour, dtype=str)
    rating_df = pd.read_csv(csv_url_rating, dtype=str)

    # Bersihkan spasi
    tour_df = tour_df.apply(lambda col: col.str.strip() if col.dtype == 'object' else col)
    rating_df = rating_df.apply(lambda col: col.str.strip() if col.dtype == 'object' else col)

    # Konversi tipe data agar konsisten
    rating_df['User_Id'] = rating_df['User_Id'].astype(float).astype(int).astype(str)
    rating_df['Place_Id'] = rating_df['Place_Id'].astype(float).astype(int).astype(str)
    tour_df['Place_Id'] = tour_df['Place_Id'].astype(float).astype(int).astype(str)

    rating_df['Place_Ratings'] = pd.to_numeric(rating_df['Place_Ratings'], errors='coerce')

    # Hapus data kosong & duplikat
    rating_df.dropna(subset=['User_Id', 'Place_Id', 'Place_Ratings'], inplace=True)
    rating_df.drop_duplicates(inplace=True)
    tour_df.dropna(subset=['Place_Name'], inplace=True)

    return tour_df, rating_df

# Load data
tour_df, rating_df = load_data()

# Filter hanya kategori Wisata
wisata_df = tour_df[tour_df['Category'] == 'Wisata']

# Buat user-item matrix (rating)
user_item_matrix = rating_df.pivot_table(index='User_Id', columns='Place_Id', values='Place_Ratings').fillna(0)

# Buat similarity matriks antar tempat wisata (item-based)
# Kita gunakan cosine similarity antar kolom (Place_Id)
cosine_sim = cosine_similarity(user_item_matrix.T)
cosine_sim_df = pd.DataFrame(cosine_sim, index=user_item_matrix.columns, columns=user_item_matrix.columns)

# Fungsi rekomendasi berdasar item terpilih
def recommend_similar_places(place_id, top_n=5):
    if place_id not in cosine_sim_df.columns:
        return pd.DataFrame()  # kosong kalau id gak ada
    
    sim_scores = cosine_sim_df[place_id].drop(place_id)  # hapus self similarity
    top_similar = sim_scores.sort_values(ascending=False).head(top_n).index

    # Gabungkan dengan info wisata
    rekom = wisata_df[wisata_df['Place_Id'].isin(top_similar)][['Place_Name', 'Category', 'Rating']]
    rekom = rekom.copy()
    rekom['Similarity_Score'] = sim_scores.loc[top_similar].values
    rekom = rekom.sort_values(by='Similarity_Score', ascending=False)
    return rekom

# Streamlit UI
st.title("Rekomendasi Tempat Wisata Berdasarkan Tempat yang Dipilih")

# Dropdown pilih tempat wisata kategori Wisata
place_name_list = wisata_df['Place_Name'].sort_values().tolist()
selected_place = st.selectbox("Pilih Tempat Wisata:", place_name_list)

if selected_place:
    place_id = wisata_df[wisata_df['Place_Name'] == selected_place]['Place_Id'].values[0]
    st.write(f"Tempat wisata yang dipilih: **{selected_place}**")

    rekomendasi = recommend_similar_places(place_id, top_n=5)
    if rekomendasi.empty:
        st.info("Maaf, tidak ada rekomendasi yang tersedia untuk tempat ini.")
    else:
        st.subheader("Rekomendasi Tempat Wisata Serupa:")
        st.dataframe(rekomendasi.reset_index(drop=True))

        # Visualisasi sederhana similarity score
        st.bar_chart(rekomendasi.set_index('Place_Name')['Similarity_Score'])
