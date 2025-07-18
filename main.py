import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Rekomendasi Wisata Jogja", layout="wide")

# ============================
# 1. Fungsi Load dari Drive
# ============================
def load_csv_from_drive(file_id):
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = requests.get(url)
    if response.status_code != 200:
        st.error("❌ Gagal mengunduh file.")
        return pd.DataFrame()
    return pd.read_csv(io.StringIO(response.content.decode('utf-8')))

# ============================
# 2. Load Dataset
# ============================
tour_csv_id = '11hQi3aqQkq5m2567jl7Ux1klXShLnYox'
rating_csv_id = '14Bke4--cJi6bVrQng8HlpFihOFOPhGZJ'
tour_df = load_csv_from_drive(tour_csv_id)
rating_df = load_csv_from_drive(rating_csv_id)

# ============================
# 3. Preprocessing
# ============================
tour_df.dropna(subset=['Place_Id', 'Place_Name'], inplace=True)
rating_df.dropna(subset=['User_Id', 'Place_Id', 'Place_Ratings'], inplace=True)

tour_df['Place_Id'] = tour_df['Place_Id'].astype(int).astype(str)
rating_df['Place_Id'] = rating_df['Place_Id'].astype(int).astype(str)
rating_df['User_Id'] = rating_df['User_Id'].astype(str)

# ============================
# 4. Similarity Matrix
# ============================
rating_matrix = rating_df.pivot_table(index='Place_Id', columns='User_Id', values='Place_Ratings').fillna(0)
item_similarity = cosine_similarity(rating_matrix)
item_similarity_df = pd.DataFrame(item_similarity, index=rating_matrix.index, columns=rating_matrix.index)

# ============================
# 5. Rekomendasi Function
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
# 6. Sidebar + Form Submit
# ============================
st.sidebar.header("Pilih Tempat Wisata")
with st.sidebar.form(key='form_rekomendasi'):
    place_names = sorted(tour_df['Place_Name'].unique())
    selected_place = st.selectbox("Nama Tempat", place_names)
    cari = st.form_submit_button("Cari Rekomendasi")

# ============================
# 7. Output
# ============================
st.title("Sistem Rekomendasi Tempat Wisata di Yogyakarta")

if cari:
    rekomendasi_df, origin_place = get_recommendation_by_name(selected_place)

    if origin_place is not None:
        st.markdown(f"Rekomendasi Mirip dengan: **{origin_place['Place_Name']}**")
        st.caption(f"Kategori: {origin_place['Category']} | Kota: {origin_place['City']}")

    if not rekomendasi_df.empty:
        st.markdown("Rekomendasi Wisata:")
        cols = st.columns(5)

        # Gunakan 1 gambar default untuk semua tempat
        image_url = "https://raw.githubusercontent.com/abimanyuprimarendra/PIDB/main/yk.jpg"

        # CSS Styling
        card_style = """
            background-color: #ffffff;
            border-radius: 15px;
            padding: 16px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            height: 400px;
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
            align-items: flex-start;
            text-align: left;
        """
        image_style = "width: 100%; height: 150px; object-fit: cover; border-radius: 12px; margin-bottom: 12px;"
        content_wrapper = "width: 100%; display: flex; flex-direction: column; gap: 4px;"
        title_style = "font-weight: bold; font-size: 16px; color: #222222; margin: 0;"
        text_style = "font-size: 13px; color: #444444; margin: 0; line-height: 1.4;"

        # Loop rekomendasi dan tampilkan dalam card
        for idx, (_, row) in enumerate(rekomendasi_df.iterrows()):
            with cols[idx]:
                place_name = row['Place_Name']
                category = row['Category']
                rating = row['Rating']
                description = row.get('Description', 'Tidak ada deskripsi')
                truncated_description = (description[:120] + "....") if len(description) > 120 else description

                st.markdown(f"""
                    <div style="{card_style}">
                        <img src="{image_url}" style="{image_style}" />
                        <div style="{content_wrapper}">
                            <p style="{title_style}">{place_name}</p>
                            <p style="{text_style}"><b>Kategori:</b> {category}</p>
                            <p style="{text_style}"><b>Rating:</b> {rating}</p>
                            <p style="{text_style}">{truncated_description}</p>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

    elif origin_place is None:
        st.warning("Tempat wisata tidak ditemukan.")
    else:
        st.info("Tidak ada rekomendasi ditemukan.")
