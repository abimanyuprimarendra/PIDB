import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(layout="wide")
st.title("🎯 Sistem Rekomendasi Tempat Wisata Yogyakarta")

# Load CSV dari Google Drive
def load_csv_from_drive(file_id):
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = requests.get(url)
    if response.status_code != 200:
        st.error("Gagal mengunduh file dari Google Drive.")
        return pd.DataFrame()
    return pd.read_csv(io.StringIO(response.content.decode('utf-8')))

# ID file dataset
tour_csv_id = '11hQi3aqQkq5m2567jl7Ux1klXShLnYox'
rating_csv_id = '14Bke4--cJi6bVrQng8HlpFihOFOPhGZJ'

# Load data
tours_df = load_csv_from_drive(tour_csv_id)
ratings_df = load_csv_from_drive(rating_csv_id)
if tours_df.empty or ratings_df.empty:
    st.stop()

# Preprocessing
pivot_table = ratings_df.pivot_table(index='User_Id', columns='Place_Id', values='Place_Ratings').fillna(0)
item_similarity_matrix = cosine_similarity(pivot_table.T)
item_similarity_df = pd.DataFrame(item_similarity_matrix, index=pivot_table.columns, columns=pivot_table.columns)
place_names = tours_df.set_index('Place_Id')['Place_Name'].to_dict()
place_name_to_id = {v: k for k, v in place_names.items()}

# Fungsi rekomendasi
def recommend_similar_places_by_category(place_id, similarity_df, tours_df, top_n=5):
    if place_id not in similarity_df or place_id not in tours_df['Place_Id'].values:
        return pd.DataFrame()
    selected_category = tours_df.loc[tours_df['Place_Id'] == place_id, 'Category'].values[0]
    sim_scores = similarity_df[place_id].sort_values(ascending=False).drop(index=place_id)
    candidate_ids = sim_scores.index
    candidate_df = tours_df[tours_df['Place_Id'].isin(candidate_ids)]
    filtered_df = candidate_df[candidate_df['Category'] == selected_category]
    return filtered_df[['Place_Name', 'Category', 'City', 'Description', 'Price']].head(top_n).reset_index(drop=True)

# Sidebar
st.sidebar.title("📌 Pilih Tempat Wisata")
selected_place_name = st.sidebar.selectbox("Nama Tempat", [""] + list(place_name_to_id.keys()))
cari_button = st.sidebar.button("🔍 Cari Rekomendasi")

# CSS: layout horizontal card dalam kolom
st.markdown("""
<style>
.card {
    background-color: white;
    border-radius: 10px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    padding: 12px;
    display: flex;
    flex-direction: column;
    align-items: center;
    height: 100%;
    text-align: center;
}
.card img {
    width: 100%;
    border-radius: 8px;
    object-fit: cover;
    margin-bottom: 10px;
}
.card-title {
    font-weight: bold;
    font-size: 16px;
    margin-bottom: 6px;
}
.card-body {
    font-size: 13px;
    color: #444;
}
</style>
""", unsafe_allow_html=True)

# Jika user klik tombol cari
if cari_button and selected_place_name != "":
    selected_place_id = place_name_to_id[selected_place_name]
    recs = recommend_similar_places_by_category(selected_place_id, item_similarity_df, tours_df, top_n=5)

    st.markdown(f"### ✨ 5 Tempat Serupa **'{selected_place_name}'**")

    if recs.empty:
        st.info("Tidak ditemukan rekomendasi.")
    else:
        cols = st.columns(5)  # 5 kolom sejajar
        for i, row in recs.iterrows():
            with cols[i]:
                st.markdown(f"""
                <div class="card">
                    <img src="https://raw.githubusercontent.com/abimanyuprimarendra/PIDB/main/yk.jpg" alt="img">
                    <div class="card-title">{row['Place_Name']}</div>
                    <div class="card-body">
                        <b>Kategori:</b> {row['Category']}<br>
                        <b>Kota:</b> {row['City']}<br>
                        <b>Harga:</b> Rp{int(row['Price']):,}<br>
                        <div style='margin-top:6px'>{row['Description'][:100]}...</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
else:
    st.empty()
