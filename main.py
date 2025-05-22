# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from heapq import nlargest
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# ========== LOAD DATA ==========
@st.cache_data
def load_data():
    tour_df = pd.read_csv(
        '/content/drive/MyDrive/Semester 6/PIDB JURNAL NON REG SCIENTIST/DATASET PARIWISATA YOGYAKARTA/Dataset Wisata Jogja.csv',
        dtype=str,
        thousands='.'
    )
    rating_df = pd.read_csv(
        '/content/drive/MyDrive/Semester 6/PIDB JURNAL NON REG SCIENTIST/DATASET PARIWISATA YOGYAKARTA/Dataset Rating Wisata Jogja.csv',
        dtype=str
    )
    return tour_df, rating_df

tour_df, rating_df = load_data()

# ========== DATA CLEANING ==========
tour_df = tour_df.apply(lambda col: col.str.strip() if col.dtype == 'object' else col)
rating_df = rating_df.apply(lambda col: col.str.strip() if col.dtype == 'object' else col)

rating_df['User_Id'] = rating_df['User_Id'].astype(float).astype(int).astype(str)
rating_df['Place_Id'] = rating_df['Place_Id'].astype(float).astype(int).astype(str)
tour_df['Place_Id'] = tour_df['Place_Id'].astype(float).astype(int).astype(str)

rating_df['Place_Ratings'] = pd.to_numeric(rating_df['Place_Ratings'], errors='coerce')
rating_df.dropna(subset=['User_Id', 'Place_Id', 'Place_Ratings'], inplace=True)
rating_df.drop_duplicates(inplace=True)
tour_df.dropna(subset=['Place_Name'], inplace=True)

# ========== USER-ITEM MATRIX ==========
user_item_matrix = rating_df.pivot_table(index='User_Id', columns='Place_Id', values='Place_Ratings')
user_item_filled = user_item_matrix.fillna(0)

# ========== SIMILARITY & CACHE ==========
item_similarity = user_item_matrix.corr(method='pearson').fillna(0)
mean_ratings_dict = user_item_matrix.mean().to_dict()

def precompute_top_k_neighbors(item_similarity, k=6):
    top_k_neighbors_dict = {}
    for place in item_similarity.columns:
        sim_scores = item_similarity[place]
        filtered_sim = sim_scores[sim_scores > 0]
        top_k = nlargest(k, filtered_sim.items(), key=lambda x: x[1])
        top_k_neighbors_dict[place] = dict(top_k)
    return top_k_neighbors_dict

top_k_neighbors = precompute_top_k_neighbors(item_similarity, k=6)

def predict_rating_fast(user_id, place_id):
    if place_id not in user_item_matrix.columns or user_id not in user_item_matrix.index:
        return np.nan
    user_ratings = user_item_matrix.loc[user_id]
    neighbors = top_k_neighbors.get(place_id, {})
    rated_neighbors = {item: sim for item, sim in neighbors.items() if not np.isnan(user_ratings.get(item, np.nan))}
    if len(rated_neighbors) == 0:
        return np.nan
    mean_target = mean_ratings_dict.get(place_id, 0)
    mean_neighbors = user_item_matrix[list(rated_neighbors.keys())].mean()
    adjusted_ratings = user_ratings[list(rated_neighbors.keys())] - mean_neighbors
    sim_scores = np.array(list(rated_neighbors.values()))
    numerator = np.dot(adjusted_ratings, sim_scores)
    denominator = np.sum(np.abs(sim_scores))
    if denominator == 0:
        return np.nan
    pred_rating = mean_target + (numerator / denominator)
    return pred_rating

def recommend_final(user_id, k=6, n_recommendations=5):
    user_id = str(user_id).strip()
    if user_id not in user_item_matrix.index:
        return f"User {user_id} tidak ditemukan di data rating."
    unrated_places = user_item_matrix.loc[user_id][user_item_matrix.loc[user_id].isna()].index
    pred_ratings = []
    for place in unrated_places:
        rating = predict_rating_fast(user_id, place)
        if not np.isnan(rating):
            pred_ratings.append((place, rating))
    if not pred_ratings:
        return f"Tidak ada rekomendasi yang dapat diberikan untuk user {user_id}."
    sorted_ratings = sorted(pred_ratings, key=lambda x: x[1], reverse=True)
    top_places = pd.DataFrame(sorted_ratings[:50], columns=['Place_Id', 'Predicted_Rating'])
    rekomendasi = pd.merge(top_places, tour_df[['Place_Id', 'Place_Name', 'Category', 'Rating']], on='Place_Id', how='left')
    user_rated = rating_df[rating_df['User_Id'] == str(user_id)]
    favorite_categories = user_rated.merge(tour_df[['Place_Id', 'Category']], on='Place_Id')
    favorite_category_counts = favorite_categories['Category'].value_counts()
    if favorite_category_counts.empty:
        final_recommendation = rekomendasi.head(n_recommendations)
    else:
        fav_cats = favorite_category_counts.index.tolist()
        rekomendasi['Category_Fav'] = rekomendasi['Category'].apply(lambda x: x in fav_cats)
        rekomendasi = rekomendasi.sort_values(by=['Category_Fav', 'Predicted_Rating'], ascending=[False, False])
        final_recommendation = rekomendasi.head(n_recommendations)
    return final_recommendation.reset_index(drop=True)

def evaluate_model(k=6):
    y_true, y_pred = [], []
    def predict_row(row):
        user_id, place_id, true_rating = row['User_Id'], row['Place_Id'], row['Place_Ratings']
        pred_rating = predict_rating_fast(user_id, place_id)
        return true_rating, pred_rating
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(tqdm(executor.map(predict_row, rating_df.to_dict('records')), total=len(rating_df), desc="Evaluasi Model"))
    for true_rating, pred_rating in results:
        if not np.isnan(pred_rating):
            y_true.append(true_rating)
            y_pred.append(pred_rating)
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return mae, rmse

# ========== STREAMLIT INTERFACE ==========
st.title("📍 Rekomendasi Tempat Wisata Yogyakarta")
user_id_input = st.text_input("Masukkan User ID", value="2")

if st.button("Tampilkan Rekomendasi"):
    with st.spinner("Mencari rekomendasi..."):
        hasil = recommend_final(user_id=user_id_input, k=6, n_recommendations=5)
        st.subheader(f"Hasil Rekomendasi untuk User {user_id_input}:")
        st.dataframe(hasil)

st.subheader("📊 Evaluasi Model dengan K Tetangga Berbeda")
neighbors_list = [4, 6, 8]
mae_list, rmse_list = [], []

for k_val in neighbors_list:
    top_k_neighbors = precompute_top_k_neighbors(item_similarity, k=k_val)
    mae_k, rmse_k = evaluate_model(k=k_val)
    mae_list.append(mae_k)
    rmse_list.append(rmse_k)

fig, ax = plt.subplots(1, 2, figsize=(12, 4))

ax[0].plot(neighbors_list, mae_list, marker='o', label='MAE')
ax[0].set_title('MAE vs Jumlah Tetangga')
ax[0].set_xlabel('Top N Neighbors')
ax[0].set_ylabel('MAE')
ax[0].grid(True)

ax[1].plot(neighbors_list, rmse_list, marker='s', color='green', label='RMSE')
ax[1].set_title('RMSE vs Jumlah Tetangga')
ax[1].set_xlabel('Top N Neighbors')
ax[1].set_ylabel('RMSE')
ax[1].grid(True)

st.pyplot(fig)
