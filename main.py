import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from heapq import nlargest
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import time

# ---------- LOAD DATASET ----------
@st.cache_data(show_spinner=True)
def load_data():
    csv_url_tour = "https://drive.google.com/uc?id=1toXFdx4bIbDevyPSmEdbs2gG3PR9iYI-"
    csv_url_rating = "https://drive.google.com/uc?id=1NUbzdY_ZNVI2Gc9avZaTvQNT6gp5tc4y"

    tour_df = pd.read_csv(csv_url_tour, dtype=str)
    rating_df = pd.read_csv(csv_url_rating, dtype=str)
    
    # Bersihkan data
    tour_df = tour_df.apply(lambda col: col.str.strip() if col.dtype == 'object' else col)
    rating_df = rating_df.apply(lambda col: col.str.strip() if col.dtype == 'object' else col)
    
    rating_df['User_Id'] = rating_df['User_Id'].astype(float).astype(int).astype(str)
    rating_df['Place_Id'] = rating_df['Place_Id'].astype(float).astype(int).astype(str)
    tour_df['Place_Id'] = tour_df['Place_Id'].astype(float).astype(int).astype(str)
    
    rating_df['Place_Ratings'] = pd.to_numeric(rating_df['Place_Ratings'], errors='coerce')
    rating_df.dropna(subset=['User_Id', 'Place_Id', 'Place_Ratings'], inplace=True)
    rating_df.drop_duplicates(inplace=True)
    tour_df.dropna(subset=['Place_Name'], inplace=True)
    
    return tour_df, rating_df

# ---------- PREPARASI DATA & METODE PREDIKSI ----------
@st.cache_data(show_spinner=False)
def prepare_data(tour_df, rating_df):
    user_item_matrix = rating_df.pivot_table(index='User_Id', columns='Place_Id', values='Place_Ratings')
    user_item_filled = user_item_matrix.fillna(0)
    cosine_sim_matrix = pd.DataFrame(
        cosine_similarity(user_item_filled.T),
        index=user_item_matrix.columns,
        columns=user_item_matrix.columns
    )
    item_similarity = user_item_matrix.corr(method='pearson')
    item_similarity_filled = item_similarity.fillna(0)
    mean_ratings_dict = user_item_matrix.mean().to_dict()
    
    return user_item_matrix, item_similarity_filled, mean_ratings_dict, tour_df, rating_df

def precompute_top_k_neighbors(item_similarity, k=6):
    top_k_neighbors_dict = {}
    for place in item_similarity.columns:
        sim_scores = item_similarity[place]
        filtered_sim = sim_scores[sim_scores > 0]
        top_k = nlargest(k, filtered_sim.items(), key=lambda x: x[1])
        top_k_neighbors_dict[place] = dict(top_k)
    return top_k_neighbors_dict

def predict_rating_fast(user_id, place_id, user_item_matrix, mean_ratings_dict, top_k_neighbors):
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

def evaluate_model(user_item_matrix, rating_df, mean_ratings_dict, top_k_neighbors):
    y_true, y_pred = [], []
    start_time = time.time()

    def predict_row(row):
        user_id, place_id, true_rating = row['User_Id'], row['Place_Id'], row['Place_Ratings']
        pred_rating = predict_rating_fast(user_id, place_id, user_item_matrix, mean_ratings_dict, top_k_neighbors)
        return true_rating, pred_rating

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(tqdm(executor.map(predict_row, rating_df.to_dict('records')), total=len(rating_df), desc="Evaluasi Model"))

    for true_rating, pred_rating in results:
        if not np.isnan(pred_rating):
            y_true.append(true_rating)
            y_pred.append(pred_rating)

    elapsed_time = (time.time() - start_time) * 1000
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    return mae, rmse, elapsed_time

# ========== STREAMLIT UI ==========

st.title("Sistem Rekomendasi Tempat Wisata Yogyakarta")

with st.spinner('Loading data...'):
    tour_df, rating_df = load_data()
    user_item_matrix, item_similarity_filled, mean_ratings_dict, tour_df, rating_df = prepare_data(tour_df, rating_df)

k_values = st.sidebar.multiselect("Pilih nilai k (jumlah tetangga):", options=[2, 4, 6, 8, 10], default=[4, 6, 8])

if st.button("Evaluasi Model dan Visualisasi"):
    mae_list, rmse_list, time_list = [], [], []
    
    for k in k_values:
        top_k_neighbors = precompute_top_k_neighbors(item_similarity_filled, k=k)
        mae, rmse, waktu = evaluate_model(user_item_matrix, rating_df, mean_ratings_dict, top_k_neighbors)
        mae_list.append(mae)
        rmse_list.append(rmse)
        time_list.append(waktu)
    
    # Plot visualisasi
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    
    axs[0].plot(k_values, mae_list, marker='o', color='blue')
    axs[0].set_title("MAE vs Jumlah Tetangga (k)")
    axs[0].set_xlabel("Jumlah Tetangga (k)")
    axs[0].set_ylabel("MAE")
    axs[0].grid(True)
    
    axs[1].plot(k_values, rmse_list, marker='s', color='green')
    axs[1].set_title("RMSE vs Jumlah Tetangga (k)")
    axs[1].set_xlabel("Jumlah Tetangga (k)")
    axs[1].set_ylabel("RMSE")
    axs[1].grid(True)
    
    axs[2].plot(k_values, time_list, marker='^', color='orange')
    axs[2].set_title("Waktu Prediksi (ms) vs Jumlah Tetangga (k)")
    axs[2].set_xlabel("Jumlah Tetangga (k)")
    axs[2].set_ylabel("Waktu Prediksi (ms)")
    axs[2].grid(True)
    
    st.pyplot(fig)
    
    eval_df = pd.DataFrame({
        'Jumlah Tetangga (k)': k_values,
        'MAE': mae_list,
        'RMSE': rmse_list,
        'Waktu Prediksi (ms)': time_list
    })
    st.dataframe(eval_df)
