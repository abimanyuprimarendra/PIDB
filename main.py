import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from heapq import nlargest
from concurrent.futures import ThreadPoolExecutor
import streamlit as st

# ========== LOAD DATA FUNCTION ==========
def load_data():
    csv_url_tour = "https://drive.google.com/uc?id=1toXFdx4bIbDevyPSmEdbs2gG3PR9iYI-"
    csv_url_rating = "https://drive.google.com/uc?id=1NUbzdY_ZNVI2Gc9avZaTvQNT6gp5tc4y"

    tour_df = pd.read_csv(csv_url_tour, dtype=str)
    rating_df = pd.read_csv(csv_url_rating, dtype=str)

    # Bersihkan spasi di kolom string
    tour_df = tour_df.apply(lambda col: col.str.strip() if col.dtype == 'object' else col)
    rating_df = rating_df.apply(lambda col: col.str.strip() if col.dtype == 'object' else col)

    # Konversi tipe data untuk ID agar konsisten
    rating_df['User_Id'] = rating_df['User_Id'].astype(float).astype(int).astype(str)
    rating_df['Place_Id'] = rating_df['Place_Id'].astype(float).astype(int).astype(str)
    tour_df['Place_Id'] = tour_df['Place_Id'].astype(float).astype(int).astype(str)

    rating_df['Place_Ratings'] = pd.to_numeric(rating_df['Place_Ratings'], errors='coerce')

    # Hapus data yang kosong dan duplikat
    rating_df.dropna(subset=['User_Id', 'Place_Id', 'Place_Ratings'], inplace=True)
    rating_df.drop_duplicates(inplace=True)
    tour_df.dropna(subset=['Place_Name'], inplace=True)

    return tour_df, rating_df

# ========== DATA LOADING ==========
tour_df, rating_df = load_data()

# ========== DATA PREPARATION ==========
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

def precompute_top_k_neighbors(item_similarity, k=6):
    top_k_neighbors_dict = {}
    for place in item_similarity.columns:
        sim_scores = item_similarity[place]
        filtered_sim = sim_scores[sim_scores > 0]
        top_k = nlargest(k, filtered_sim.items(), key=lambda x: x[1])
        top_k_neighbors_dict[place] = dict(top_k)
    return top_k_neighbors_dict

top_k_neighbors = precompute_top_k_neighbors(item_similarity_filled, k=6)

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

def recommend_places_optimized(user_id, k=6, n_recommendations=5):
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
    top_places = pd.DataFrame(sorted_ratings[:n_recommendations], columns=['Place_Id', 'Predicted_Rating'])

    if 'Place_Id' in tour_df.columns:
        rekomendasi = pd.merge(top_places, tour_df[['Place_Id', 'Place_Name', 'Category', 'Rating']], on='Place_Id', how='left')
    else:
        rekomendasi = top_places

    return rekomendasi

def recommend_final(user_id, k=6, n_recommendations=5):
    rekomendasi = recommend_places_optimized(user_id, k=k, n_recommendations=50)

    if isinstance(rekomendasi, str):
        return rekomendasi

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

def recommend_similar_place(place_id, n=5):
    if place_id not in item_similarity.columns:
        return f"Tempat dengan ID {place_id} tidak ditemukan."
    similar_places = item_similarity[place_id].drop(labels=[place_id]).sort_values(ascending=False).head(n)
    df_similar = pd.DataFrame(similar_places).reset_index()
    df_similar.columns = ['Place_Id', 'Similarity']
    df_merged = pd.merge(df_similar, tour_df[['Place_Id', 'Place_Name', 'Category', 'Rating']], on='Place_Id', how='left')
    return df_merged

def evaluate_model(k=6):
    y_true, y_pred = [], []
    start_time = time.time()

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

    elapsed_time = (time.time() - start_time) * 1000
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    return mae, rmse, elapsed_time

# ========== STREAMLIT UI ==========
st.title("📍 Rekomendasi Tempat Wisata Yogyakarta")

user_id_input = st.text_input("Masukkan User ID untuk rekomendasi wisata:", value="2")
if st.button("Dapatkan Rekomendasi"):
    rekom = recommend_final(user_id=user_id_input, k=6, n_recommendations=5)
    if isinstance(rekom, str):
        st.warning(rekom)
    else:
        st.dataframe(rekom)

st.markdown("---")

st.subheader("🔍 Eksplorasi Tempat Serupa Berdasarkan Wisata")
place_options = tour_df[['Place_Id', 'Place_Name']].drop_duplicates().sort_values('Place_Name')
place_name_selected = st.selectbox("Pilih Tempat Wisata", place_options['Place_Name'].tolist())

if st.button("Lihat Tempat Serupa"):
    selected_place_id = place_options[place_options['Place_Name'] == place_name_selected]['Place_Id'].values[0]
    with st.spinner("Mencari tempat wisata serupa..."):
        similar_result = recommend_similar_place(selected_place_id, n=5)
        st.dataframe(similar_result)

st.markdown("---")

neighbors_list = [4, 6, 8]
mae_list, rmse_list, time_list = [], [], []

for k_val in neighbors_list:
    top_k_neighbors = precompute_top_k_neighbors(item_similarity_filled, k=k_val)
    mae_k, rmse_k, time_k = evaluate_model(k=k_val)
    mae_list.append(mae_k)
    rmse_list.append(rmse_k)
    time_list.append(time_k)

st.subheader("📊 Evaluasi Model")
fig, axs = plt.subplots(1, 3, figsize=(14, 5))

axs[0].plot(neighbors_list, mae_list, marker='o')
axs[0].set_title('MAE vs Jumlah Tetangga')
axs[0].set_xlabel('Top N Neighbors')
axs[0].set_ylabel('MAE')
axs[0].grid(True)

axs[1].plot(neighbors_list, rmse_list, marker='s', color='green')
axs[1].set_title('RMSE vs Jumlah Tetangga')
axs[1].set_xlabel('Top N Neighbors')
axs[1].set_ylabel('RMSE')
axs[1].grid(True)

axs[2].plot(neighbors_list, time_list, marker='^', color='orange')
axs[2].set_title('Waktu Prediksi vs Top N')
axs[2].set_xlabel('Top N Neighbors')
axs[2].set_ylabel('Waktu (ms)')
axs[2].grid(True)

st.pyplot(fig)
