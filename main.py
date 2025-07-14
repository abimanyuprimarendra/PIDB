import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(layout="wide")
st.title("Sistem Rekomendasi Tempat Wisata Yogyakarta")

# Fungsi untuk load CSV dari Google Drive
def load_csv_from_drive(file_id):
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = requests.get(url)
    if response.status_code != 200:
        st.error("Gagal mengunduh file dari Google Drive.")
        return pd.DataFrame()
    return pd.read_csv(io.StringIO(response.content.decode('utf-8')))

# Fungsi reverse geocoding
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

# File ID Google Drive
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

# Fungsi rekomendasi
def recommend_similar_places_by_category(place_id, similarity_df, tours_df, top_n=5):
    if place_id not in similarity_df or place_id not in tours_df['Place_Id'].values:
        return pd.DataFrame()
    selected_category = tours_df.loc[tours_df['Place_Id'] == place_id, 'Category'].values[0]
    sim_scores = similarity_df[place_id].sort_values(ascending=False)
    candidate_ids = sim_scores.iloc[1:top_n*3].index
    candidate_df = tours_df[tours_df['Place_Id'].isin(candidate_ids)]
    filtered_df = candidate_df[candidate_df['Category'] == selected_category]
    return filtered_df[['Place_Id', 'Place_Name', 'Category', 'City', 'Description', 'Price']].head(top_n).reset_index(drop=True)

# Sidebar pencarian
st.sidebar.title("Cari Tempat Wisata")
place_name_to_id = {v: k for k, v in place_names.items()}
selected_place_name = st.sidebar.selectbox("Pilih Tempat Wisata", list(place_name_to_id.keys()))
cari_button = st.sidebar.button("Cari Rekomendasi")

selected_place_id = place_name_to_id[selected_place_name]

# Tampilan tempat terpilih
st.header(f"Tempat Wisata: {selected_place_name}")
selected_row = tours_df[tours_df['Place_Id'] == selected_place_id].iloc[0]
st.markdown(f"**Kategori:** {selected_row['Category']}  \n**Kota:** {selected_row['City']}")
if 'Description' in selected_row:
    st.markdown(f"**Deskripsi:** {selected_row['Description']}")
if 'Price' in selected_row:
    st.markdown(f"**Harga:** Rp{int(selected_row['Price']):,}")

# Rekomendasi hanya jika tombol ditekan
if cari_button:
    st.subheader("Rekomendasi Tempat Serupa")
    recs = recommend_similar_places_by_category(selected_place_id, item_similarity_df, tours_df, top_n=5)

    if recs.empty:
        st.info("Tidak ditemukan rekomendasi.")
    else:
        for _, row in recs.iterrows():
            with st.container():
                col1, col2 = st.columns([1, 4])
                with col1:
                    st.image("https://static.thenounproject.com/png/3470437-200.png", width=80)  # placeholder ikon
                with col2:
                    st.subheader(row['Place_Name'])
                    st.markdown(f"**Kategori:** {row['Category']}  \n**Kota:** {row['City']}")
                    if pd.notna(row['Description']):
                        st.markdown(f"**Deskripsi:** {row['Description']}")
                    if pd.notna(row['Price']):
                        st.markdown(f"**Harga:** Rp{int(row['Price']):,}")
            st.markdown("---")
