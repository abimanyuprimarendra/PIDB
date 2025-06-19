import streamlit as st
import pandas as pd
import numpy as np
import gdown
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# Fungsi membersihkan harga
def clean_price(x):
    if pd.isna(x): return np.nan
    return float(str(x).replace('Rp', '').replace('.', '').replace(',', '').strip())

@st.cache_data
def load_data_from_drive():
    csv_url = "https://drive.google.com/uc?id=1F4LiTAs79DDimrQgKCUi1HqHQ-HmIYEj"
    data = pd.read_csv(csv_url)
    return data


    # 1. DATA CLEANING
    df.drop(columns=['image'], inplace=True, errors='ignore')
    df.replace(['-', 'null', 'NaN', ''], np.nan, inplace=True)
    df.drop_duplicates(inplace=True)

    df['htm_weekday'] = df['htm_weekday'].apply(clean_price)
    df['htm_weekend'] = df['htm_weekend'].apply(clean_price)

    # 2. TYPE CONVERSION
    num_cols = ['vote_average', 'vote_count', 'latitude', 'longitude']
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')

    # 3. MISSING VALUE HANDLING
    df['htm_weekday'] = df['htm_weekday'].fillna(df['htm_weekday'].median())
    df['htm_weekend'] = df['htm_weekend'].fillna(df['htm_weekend'].median())
    df['vote_average'] = df['vote_average'].fillna(0)
    df['vote_count'] = df['vote_count'].fillna(0)
    df['latitude'] = df['latitude'].fillna(0)
    df['longitude'] = df['longitude'].fillna(0)
    df['description'] = df['description'].fillna('')

    # 4. CATEGORICAL ENCODING
    df['type_encoded'] = df['type'].astype('category').cat.codes

    return df

# Fungsi rekomendasi
def get_recommendations(df, nama_wisata, top_n=5):
    features = ['vote_average', 'vote_count', 'htm_weekday', 'htm_weekend']
    scaler = StandardScaler()
    item_features_scaled = scaler.fit_transform(df[features])

    similarity_matrix = cosine_similarity(item_features_scaled)
    similarity_df = pd.DataFrame(similarity_matrix, index=df['nama'], columns=df['nama'])

    if nama_wisata not in similarity_df.index:
        return []

    similar = similarity_df[nama_wisata].sort_values(ascending=False)
    return similar.iloc[1:top_n+1]

# Tampilan UI Streamlit
st.title('📍 Rekomendasi Tempat Wisata di Yogyakarta')
df = load_data()

nama_wisata_list = df['nama'].dropna().unique()
selected_wisata = st.selectbox("Pilih tempat wisata:", nama_wisata_list)

top_n = st.slider("Jumlah rekomendasi", 1, 10, 5)

if st.button("Tampilkan Rekomendasi"):
    rekomendasi = get_recommendations(df, selected_wisata, top_n)
    if rekomendasi is not None and not rekomendasi.empty:
        st.subheader(f"Rekomendasi mirip dengan **{selected_wisata}**:")
        for i, (nama, skor) in enumerate(rekomendasi.items(), start=1):
            st.markdown(f"**{i}. {nama}** - Similarity Score: `{skor:.3f}`")
    else:
        st.warning("Tempat wisata tidak ditemukan atau tidak ada rekomendasi.")
