import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Rekomendasi Wisata Jogja", layout="wide")

# ============================
# Load CSV dari Drive
# ============================
def load_csv_from_drive(file_id):
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = requests.get(url)
    if response.status_code != 200:
        st.error("❌ Gagal mengunduh file.")
        return pd.DataFrame()
    return pd.read_csv(io.StringIO(response.content.decode('utf-8')))

# ============================
# Reverse Geocoding dari Koordinat
# ============================
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

# ============================
# Load Dataset
# ============================
tour_csv_id = '11hQi3aqQkq5m2567jl7Ux1klXShLnYox'
rating_csv_id = '14Bke4--cJi6bVrQng8HlpFihOFOPhGZJ'
tour_df = load_csv_from_drive(tour_csv_id)
rating_df = load_csv_from_drive(rating_csv_id)

# ============================
# Preprocessing
# ============================
tour_df.dropna(subset=['Place_Id', 'Place_Name'], inplace=True)
rating_df.dropna(subset=['User_Id', 'Place_Id', 'Place_Ratings'], inplace=True)

tour_df['Place_Id'] = tour_df['Place_Id'].astype(int).astype(str)
rating_df['Place_Id'] = rating_df['Place_Id'].astype(int).astype(str)
rating_df['User_Id'] = rating_df['User_Id'].astype(str)

# ============================
# Similarity Matrix
# ============================
rating_matrix = rating_df.pivot_table(index='Place_Id', columns='User_Id', values='Place_Ratings').fillna(0)
item_similarity = cosine_similarity(rating_matrix)
item_similarity_df = pd.DataFrame(item_similarity, index=rating_matrix.index, columns=rating_matrix.index)

# ============================
# Rekomendasi Function
# ============================
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

# ============================
# Sidebar
# ============================
st.sidebar.header("Pilih Tempat Wisata")
with st.sidebar.form(key='form_rekomendasi'):
    place_names = sorted(tour_df['Place_Name'].unique())
    selected_place = st.selectbox("Nama Tempat", place_names)
    cari = st.form_submit_button("Cari Rekomendasi")

# ============================
# Output
# ============================
st.title("Sistem Rekomendasi Tempat Wisata di Yogyakarta")

if cari:
    rekomendasi_df, origin_place = get_recommendation_by_name(selected_place)

    if origin_place is not None:
        st.markdown(f"### Rekomendasi Mirip dengan: **{origin_place['Place_Name']}**")
        st.caption(f"Kategori: {origin_place['Category']} | Kota: {origin_place['City']}")

    if not rekomendasi_df.empty:
        github_image_url = "https://raw.githubusercontent.com/abimanyuprimarendra/PIDB/main/yk.jpg"

        # Inject CSS untuk responsif layout
        st.markdown("""
        <style>
        .card-container {
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            justify-content: center;
        }

        .card {
            background-color: #f9f9f9;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.08);
            padding: 15px;
            display: flex;
            flex-direction: row;
            gap: 15px;
            width: 100%;
            max-width: 800px;
        }

        .card img {
            width: 280px;
            height: 200px;
            object-fit: cover;
            border-radius: 10px;
        }

        .card-content {
            flex: 1;
            text-align: left;
        }

        @media (max-width: 768px) {
            .card {
                flex-direction: column;
                align-items: center;
            }

            .card img {
                width: 100%;
                height: auto;
            }

            .card-content {
                text-align: center;
            }
        }
        </style>
        """, unsafe_allow_html=True)

        st.markdown('<div class="card-container">', unsafe_allow_html=True)

        for _, row in rekomendasi_df.iterrows():
            address = get_address(row['Latitude'], row['Longitude'])
            description = row.get('Description', '')

            st.markdown(f"""
            <div class="card">
                <img src="{github_image_url}">
                <div class="card-content">
                    <h3>{row['Place_Name']}</h3>
                    <p><b>Kategori:</b> {row['Category']}</p>
                    <p><b>Kota:</b> {row['City']}</p>
                    <p><b>Rating:</b> {row['Rating']}</p>
                    <p><b>Alamat:</b> {address}</p>
                    <p>{description}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    elif origin_place is None:
        st.warning("Tempat wisata tidak ditemukan.")
    else:
        st.info("Tidak ada rekomendasi ditemukan.")
