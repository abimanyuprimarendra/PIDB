import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import pydeck as pdk

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
    return result_df[['nama', 'description', 'htm_weekday', 'htm_weekend', 'vote_average', 'latitude', 'longitude', 'similarity_score']]

# UI Streamlit
st.set_page_config(page_title="Rekomendasi Wisata Jogja", layout="wide")
st.title('📍 Rekomendasi Tempat Wisata di Yogyakarta')

df = load_data_from_drive()
nama_wisata_list = df['nama'].dropna().unique()
selected_wisata = st.selectbox("🎯 Pilih tempat wisata sebagai acuan:", nama_wisata_list)
top_n = st.slider("🔢 Jumlah rekomendasi ditampilkan", 1, 10, 5)

if st.button("Tampilkan Rekomendasi"):
    rekomendasi_df = get_recommendations(df, selected_wisata, top_n)

    if not rekomendasi_df.empty:
        st.subheader(f"✅ Tempat wisata mirip dengan **{selected_wisata}**:")

        # Format harga
        rekomendasi_df['htm_weekday'] = rekomendasi_df['htm_weekday'].apply(lambda x: f"Rp {int(x):,}".replace(",", "."))
        rekomendasi_df['htm_weekend'] = rekomendasi_df['htm_weekend'].apply(lambda x: f"Rp {int(x):,}".replace(",", "."))

        # Tabel rekomendasi
        st.table(
            rekomendasi_df[['nama', 'htm_weekday', 'htm_weekend', 'vote_average', 'similarity_score']]
            .sort_values(by='similarity_score', ascending=False)
            .rename(columns={
                'nama': 'Nama Wisata',
                'htm_weekday': 'HTM Weekday',
                'htm_weekend': 'HTM Weekend',
                'vote_average': 'Rating',
                'similarity_score': 'Skor Kemiripan'
            })
            .style.format({'Skor Kemiripan': '{:.2f}', 'Rating': '{:.1f}'})
        )

        # Visualisasi Peta Pydeck
        st.subheader("🗺️ Lokasi Rekomendasi di Peta")

        layer = pdk.Layer(
            "ScatterplotLayer",
            data=rekomendasi_df,
            get_position='[longitude, latitude]',
            get_fill_color='[0, 100, 255, 160]',
            get_radius=100,
            pickable=True,
        )

        view_state = pdk.ViewState(
            latitude=rekomendasi_df['latitude'].mean(),
            longitude=rekomendasi_df['longitude'].mean(),
            zoom=11,
            pitch=0
        )

        tooltip = {
            "html": "<b>{nama}</b><br/>HTM: {htm_weekday}<br/>Rating: {vote_average}",
            "style": {
                "backgroundColor": "steelblue",
                "color": "white"
            }
        }

        st.pydeck_chart(pdk.Deck(
            map_style="mapbox://styles/mapbox/light-v9",
            initial_view_state=view_state,
            layers=[layer],
            tooltip=tooltip
        ))
    else:
        st.warning("❌ Tempat wisata tidak ditemukan atau tidak ada rekomendasi.")
