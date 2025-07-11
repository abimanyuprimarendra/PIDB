import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset (sesuaikan path dengan file lokal atau Drive-mount kamu)
@st.cache_data
def load_data():
    base_path = '/content/drive/MyDrive/Semester 6/PIDB JURNAL NON REG SCIENTIST/DATASET PARIWISATA YOGYAKARTA/'
    tours_df = pd.read_csv(base_path + 'tour.csv')
    ratings_df = pd.read_csv(base_path + 'tour_rating.csv')
    return tours_df, ratings_df

tours_df, ratings_df = load_data()

# Membuat pivot table dan matriks similarity
@st.cache_data
def prepare_similarity(ratings_df):
    pivot_table = ratings_df.pivot_table(index='User_Id', columns='Place_Id', values='Place_Ratings').fillna(0)
    similarity_matrix = cosine_similarity(pivot_table.T)
    similarity_df = pd.DataFrame(similarity_matrix, index=pivot_table.columns, columns=pivot_table.columns)
    return pivot_table, similarity_df

pivot_filled, item_similarity_df = prepare_similarity(ratings_df)

place_names = tours_df.set_index('Place_Id')['Place_Name'].to_dict()

# Fungsi prediksi rating
def predict_rating(user_id, item_id, pivot_df, similarity_df):
    if item_id not in similarity_df.columns or user_id not in pivot_df.index:
        return np.nan
    user_ratings = pivot_df.loc[user_id]
    item_similarities = similarity_df[item_id]
    rated_items = user_ratings[user_ratings > 0].index
    if len(rated_items) == 0:
        return np.nan
    sim_scores = item_similarities[rated_items]
    ratings = user_ratings[rated_items]
    if sim_scores.sum() == 0:
        return np.nan
    predicted_rating = np.dot(sim_scores, ratings) / sim_scores.sum()
    return predicted_rating

# Fungsi rekomendasi tempat serupa berdasarkan kategori
def recommend_similar_places_by_category(place_id, similarity_df, tours_df, top_n=5):
    if place_id not in similarity_df:
        return pd.DataFrame(columns=['Place_Id', 'Place_Name', 'Category', 'City'])
    if place_id not in tours_df['Place_Id'].values:
        return pd.DataFrame(columns=['Place_Id', 'Place_Name', 'Category', 'City'])
    selected_category = tours_df.loc[tours_df['Place_Id'] == place_id, 'Category'].values[0]
    sim_scores = similarity_df[place_id].sort_values(ascending=False)
    candidate_ids = sim_scores.iloc[1:top_n*3].index
    candidate_df = tours_df[tours_df['Place_Id'].isin(candidate_ids)]
    filtered_df = candidate_df[candidate_df['Category'] == selected_category]
    return filtered_df[['Place_Id', 'Place_Name', 'Category', 'City']].head(top_n).reset_index(drop=True)

# Streamlit UI
st.title("Sistem Rekomendasi Tempat Wisata Yogyakarta")

# Pilihan User_Id dan Place_Id dari data yang tersedia
user_ids = pivot_filled.index.tolist()
place_ids = list(place_names.keys())

user_id = st.selectbox("Pilih User_Id", user_ids)
place_id = st.selectbox("Pilih Place_Id (tempat wisata)", place_ids, format_func=lambda x: f"{x} - {place_names.get(x, 'Unknown')}")

# Tampilkan prediksi rating
pred_rating = predict_rating(user_id, place_id, pivot_filled, item_similarity_df)

if np.isnan(pred_rating):
    st.warning("Prediksi rating tidak tersedia untuk kombinasi User_Id dan Place_Id ini.")
else:
    st.success(f"Prediksi rating User {user_id} untuk tempat '{place_names.get(place_id)}' adalah: {pred_rating:.2f}")

# Tampilkan rekomendasi tempat serupa
st.subheader(f"Rekomendasi tempat wisata serupa berdasarkan kategori '{tours_df.loc[tours_df['Place_Id']==place_id, 'Category'].values[0]}'")

recommendations = recommend_similar_places_by_category(place_id, item_similarity_df, tours_df, top_n=5)

if recommendations.empty:
    st.info("Tidak ditemukan rekomendasi yang sesuai.")
else:
    st.dataframe(recommendations)
