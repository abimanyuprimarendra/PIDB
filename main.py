import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import folium
from streamlit_folium import st_folium

# Fungsi membersihkan harga
def clean_price(x):
    if pd.isna(x): return np.nan
    return float(str(x).replace('Rp', '').replace('.', '').replace(',', '').strip())

@st.cache_data
def load_data_from_drive():
    csv_url = "https://drive.google.com/uc?id=1F4LiTAs79DDimrQgKCUi1HqHQ-HmIYEj"
    df = pd.read_csv(csv_url)

    df.drop(columns=['image'], inplace=True, errors='ignore')
    df.replace(['-', 'null', 'NaN', ''], np.nan, inplace=True)
    df.drop_duplicates(inplace=True)

    df['htm_weekday'] = df['htm_weekday'].apply(clean_price)
    df['htm_weekend'] = df['htm_weekend'].apply(clean_price)

    num_cols = ['vote_average', 'vote_count', 'latitude', 'longitude']
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')

    df['htm_weekday'] = df['htm_weekday'].fillna(df['htm_weekday'].median())
    df['htm_weekend'] = df['htm_weekend'].fillna(df['htm_weekend'].median())
    df['vote_average'] = df['vote_average'].fillna(0)
    df['vote_count'] = df['vote_count'].fillna(0)
    df['latitude'] = df['latitude'].fillna(0)
    df['longitude'] = df['longitude'].fillna(0)
    df['description'] = df['description'].fillna('')

    df['type_encoded'] = df['type'].astype('category').cat.codes

    return df

def get_recommendations(df, nama_wisata, top_n=5):
    features = ['vote_average', 'vote_count', 'htm_weekday', 'htm_weekend']
    scaler = StandardScaler()
    item_features_scaled = scaler.fit_transform(df[features])

    similarity_matrix = cosine_similarity(item_features_scaled)
    similarity_df = pd.DataFrame(similarity_matrix, index=df['nama'], columns=df['nama'])

    if nama_wisata not in similarity_df.index:
        return pd.DataFrame()

    similar = similarity_df[nama_wisata].sort_values(ascending=False).iloc[1:top_n+1]
    result_df = df[df['nama'].isin(similar.index)].copy()
    result_df['similarity_score'] = similar.values
    return result_df[['nama', 'type', 'description', 'htm_weekday', 'htm_weekend', 'vote_average', 'latitude', 'longitude', 'similarity_score']]

# UI Streamlit
st.set_page_config(page_title="Rekomendasi Wisata Jogja", layout="wide")
st.title('Rekomendasi Tempat Wisata di Yogyakarta')

df = load_data_from_drive()

# Filter kategori dan rating
with st.sidebar:
    st.header("Filter Tambahan")
    kategori_unik = sorted(df['type'].dropna().unique())
    kategori_pilihan = st.multiselect("Pilih kategori:", kategori_unik, default=kategori_unik)

    min_rating = float(df['vote_average'].min())
    max_rating = float(df['vote_average'].max())
    rating_range = st.slider("Batas rating:", min_value=min_rating, max_value=max_rating, value=(min_rating, max_rating))

# Terapkan filter
df_filtered = df[df['type'].isin(kategori_pilihan) & df['vote_average'].between(rating_range[0], rating_range[1])]

# Input wisata
st.session_state.setdefault("last_selected", None)

nama_wisata_list = df_filtered['nama'].dropna().unique()
selected_wisata = st.selectbox("Pilih tempat wisata sebagai acuan:", nama_wisata_list, index=0)

# Reset otomatis jika wisata berubah
if selected_wisata != st.session_state["last_selected"]:
    st.session_state["show_rekomendasi"] = False
    st.session_state["last_selected"] = selected_wisata

st.session_state.setdefault("show_rekomendasi", False)

# Slider dan tombol rekomendasi
top_n = st.slider("Jumlah rekomendasi ditampilkan", 1, 10, 5)

if st.button("Tampilkan Rekomendasi"):
    st.session_state["show_rekomendasi"] = True

if st.session_state["show_rekomendasi"]:
    rekomendasi_df = get_recommendations(df_filtered, selected_wisata, top_n)

    if not rekomendasi_df.empty:
        st.subheader("Peta Lokasi Rekomendasi")
        m = folium.Map(location=[rekomendasi_df['latitude'].mean(), rekomendasi_df['longitude'].mean()], zoom_start=12)

        for _, row in rekomendasi_df.iterrows():
            popup_info = f"""
                <div style='font-size: 12px'>
                <b>{row['nama']}</b><br>
                Kategori: {row['type']}<br>
                HTM: {row['htm_weekday']}<br>
                Rating: {row['vote_average']}<br>
                Skor Kemiripan: {row['similarity_score']:.2f}<br>
                Koordinat: ({row['latitude']}, {row['longitude']})
                </div>
            """
            folium.Marker(
                location=[row['latitude'], row['longitude']],
                popup=folium.Popup(popup_info, max_width=250)
            ).add_to(m)

        st_folium(m, width=700, height=500)
    else:
        st.warning("Tempat wisata tidak ditemukan atau tidak ada rekomendasi.")
