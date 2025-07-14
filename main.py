import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Rekomendasi Wisata Jogja", layout="wide")

# ============================ LOAD CSV ============================
def load_csv_from_drive(file_id):
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = requests.get(url)
    if response.status_code != 200:
        st.error("❌ Gagal mengunduh file.")
        return pd.DataFrame()
    return pd.read_csv(io.StringIO(response.content.decode('utf-8')))

# ============================ REVERSE GEOCODING ============================
@st.cache_data(show_spinner=False)
def get_address(lat, lon):
    try:
        url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lon}"
        headers = {"User-Agent": "streamlit-wisata-yogya"}
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            return data.get("display_name", "Alamat tidak ditemukan")
        return "Alamat tidak ditemukan"
    except:
        return "Alamat tidak ditemukan"

# ============================ LOAD DATASET ============================
tour_csv_id = '11hQi3aqQkq5m2567jl7Ux1klXShLnYox'
rating_csv_id = '14Bke4--cJi6bVrQng8HlpFihOFOPhGZJ'
tour_df = load_csv_from_drive(tour_csv_id)
rating_df = load_csv_from_drive(rating_csv_id)

# ============================ PREPROCESS ============================
tour_df.dropna(subset=['Place_Id', 'Place_Name'], inplace=True)
rating_df.dropna(subset=['User_Id', 'Place_Id', 'Place_Ratings'], inplace=True)
tour_df['Place_Id'] = tour_df['Place_Id'].astype(int).astype(str)
rating_df['Place_Id'] = rating_df['Place_Id'].astype(int).astype(str)
rating_df['User_Id'] = rating_df['User_Id'].astype(str)

# ============================ SIMILARITY ============================
rating_matrix = rating_df.pivot_table(index='Place_Id', columns='User_Id', values='Place_Ratings').fillna(0)
item_similarity = cosine_similarity(rating_matrix)
item_similarity_df = pd.DataFrame(item_similarity, index=rating_matrix.index, columns=rating_matrix.index)

# ============================ RECOMMENDATION ============================
def get_recommendations(place_id, top_n=5):
    place_id = str(int(place_id))
    if place_id not in item_similarity_df.index:
        return pd.DataFrame()
    similar_scores = item_similarity_df[place_id].sort_values(ascending=False).drop(place_id)
    top_places = similar_scores.head(top_n).index.tolist()
    return tour_df[tour_df['Place_Id'].isin(top_places)]

def get_recommendation_by_name(place_name, top_n=5):
    match = tour_df[tour_df['Place_Name'].str.lower() == place_name.lower()]
    if match.empty:
        return pd.DataFrame(), None
    place_id = match['Place_Id'].values[0]
    origin = match.iloc[0]
    return get_recommendations(place_id, top_n), origin

# ============================ SIDEBAR ============================
st.sidebar.header("Pilih Tempat Wisata")
with st.sidebar.form(key='form_rekomendasi'):
    place_names = sorted(tour_df['Place_Name'].unique())
    selected_place = st.selectbox("Nama Tempat", place_names)
    cari = st.form_submit_button("Cari Rekomendasi")

# ============================ OUTPUT ============================
st.title("📍 Sistem Rekomendasi Tempat Wisata di Yogyakarta")

if cari:
    rekomendasi_df, origin_place = get_recommendation_by_name(selected_place)

    if origin_place is not None:
        st.markdown(f"### Rekomendasi Mirip dengan: **{origin_place['Place_Name']}**")
        st.caption(f"Kategori: {origin_place['Category']} | Kota: {origin_place['City']}")

    if not rekomendasi_df.empty:
        st.markdown("### 5 Rekomendasi Tempat Wisata:")

        # Styling CSS
        st.markdown("""
        <style>
        .scroll-container {
            display: flex;
            overflow-x: auto;
            gap: 20px;
            padding-bottom: 10px;
        }
        .scroll-container::-webkit-scrollbar {
            height: 8px;
        }
        .scroll-container::-webkit-scrollbar-thumb {
            background: #ccc;
            border-radius: 4px;
        }
        .card {
            background-color: #fff;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            width: 250px;
            min-width: 250px;
            flex-shrink: 0;
            display: flex;
            flex-direction: column;
            overflow: hidden;
            font-family: 'Segoe UI', sans-serif;
        }
        .card img {
            width: 100%;
            height: 150px;
            object-fit: cover;
        }
        .card-content {
            padding: 12px;
            font-size: 13px;
        }
        .card-content h4 {
            margin: 0 0 8px 0;
            font-size: 15px;
        }
        .card-content p {
            margin: 3px 0;
        }
        .description {
            font-size: 12px;
            color: #555;
            max-height: 50px;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        </style>
        """, unsafe_allow_html=True)

        st.markdown('<div class="scroll-container">', unsafe_allow_html=True)

        github_image_url = "https://raw.githubusercontent.com/abimanyuprimarendra/PIDB/main/yk.jpg"

        for _, row in rekomendasi_df.iterrows():
            address = get_address(row['Latitude'], row['Longitude'])
            description = row.get("Description", "")
            st.markdown(f"""
                <div class="card">
                    <img src="{github_image_url}">
                    <div class="card-content">
                        <h4>{row['Place_Name']}</h4>
                        <p><b>Kategori:</b> {row['Category']}</p>
                        <p><b>Kota:</b> {row['City']}</p>
                        <p><b>Rating:</b> {row['Rating']}</p>
                        <p><b>Alamat:</b> {address}</p>
                        <p class="description">{description}</p>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    elif origin_place is None:
        st.warning("Tempat wisata tidak ditemukan.")
    else:
        st.info("Tidak ada rekomendasi ditemukan.")
