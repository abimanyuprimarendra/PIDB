# Import library
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from heapq import nlargest
from concurrent.futures import ThreadPoolExecutor

# ========== DATA LOADING ==========
tour_df = pd.read_csv(
    '/content/drive/MyDrive/Semester 6/PIDB JURNAL NON REG SCIENTIST/DATASET PARIWISATA YOGYAKARTA/Dataset Wisata Jogja.csv',
    dtype=str,
    thousands='.'
)

rating_df = pd.read_csv(
    '/content/drive/MyDrive/Semester 6/PIDB JURNAL NON REG SCIENTIST/DATASET PARIWISATA YOGYAKARTA/Dataset Rating Wisata Jogja.csv',
    dtype=str
)

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

# ========== COSINE SIMILARITY MATRIX ==========
user_item_filled = user_item_matrix.fillna(0)
cosine_sim_matrix = pd.DataFrame(
    cosine_similarity(user_item_filled.T),
    index=user_item_matrix.columns,
    columns=user_item_matrix.columns
)

# ========== ITEM SIMILARITY (PEARSON) UNTUK PREDIKSI ==========
item_similarity = user_item_matrix.corr(method='pearson')
item_similarity_filled = item_similarity.fillna(0)

# ========== CACHE DATA ==========
mean_ratings_dict = user_item_matrix.mean().to_dict()

# ========== PRECOMPUTE TOP-K NEIGHBORS ==========
def precompute_top_k_neighbors(item_similarity, k=6):
    top_k_neighbors_dict = {}
    for place in item_similarity.columns:
        sim_scores = item_similarity[place]
        filtered_sim = sim_scores[sim_scores > 0]
        top_k = nlargest(k, filtered_sim.items(), key=lambda x: x[1])
        top_k_neighbors_dict[place] = dict(top_k)
    return top_k_neighbors_dict

top_k_neighbors = precompute_top_k_neighbors(item_similarity_filled, k=6)

# ========== PREDIKSI RATING ==========
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

# ========== REKOMENDASI ==========
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

# ========== REKOMENDASI FINAL ==========
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

# ========== EVALUASI ==========
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

    return mae, rmse, elapsed_time, y_true, y_pred

# ========== CONTOH PENGGUNAAN ==========
user_id = '2'
hasil_rekomendasi = recommend_final(user_id=user_id, k=6, n_recommendations=5)
print(f"\nRekomendasi final untuk User {user_id}:\n", hasil_rekomendasi)

mae, rmse, waktu, y_true, y_pred = evaluate_model(k=6)
print(f"\nEvaluasi Model (k=6):")
print(f"MAE  = {mae:.4f}")
print(f"RMSE = {rmse:.4f}")
print(f"Waktu Prediksi = {waktu:.0f} ms")

# ========== VISUALISASI ==========
neighbors_list = [4, 6, 8]
mae_list, rmse_list, time_list = [], [], []

for k_val in neighbors_list:
    top_k_neighbors = precompute_top_k_neighbors(item_similarity_filled, k=k_val)
    mae_k, rmse_k, time_k, _, _ = evaluate_model(k=k_val)
    mae_list.append(mae_k)
    rmse_list.append(rmse_k)
    time_list.append(time_k)

plt.figure(figsize=(14, 5))

plt.subplot(1, 3, 1)
plt.plot(neighbors_list, mae_list, marker='o', label='MAE')
plt.title('MAE vs Jumlah Tetangga')
plt.xlabel('Top N Neighbors')
plt.ylabel('MAE')
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(neighbors_list, rmse_list, marker='s', color='green', label='RMSE')
plt.title('RMSE vs Jumlah Tetangga')
plt.xlabel('Top N Neighbors')
plt.ylabel('RMSE')
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(neighbors_list, time_list, marker='^', color='orange', label='Waktu')
plt.title('Waktu Prediksi vs Top N')
plt.xlabel('Top N Neighbors')
plt.ylabel('Waktu (ms)')
plt.grid(True)

plt.tight_layout()
plt.show()

# === VISUALISASI TAMBAHAN ===

# 1. Distribusi Rating Aktual vs Prediksi
plt.figure(figsize=(10,6))
sns.kdeplot(y_true, label='Rating Aktual', shade=True)
sns.kdeplot(y_pred, label='Rating Prediksi', shade=True)
plt.title('Distribusi Rating Aktual vs Prediksi')
plt.xlabel('Rating')
plt.legend()
plt.show()

# 2. Heatmap Cosine Similarity Top N Item
top_n = 20
plt.figure(figsize=(12,10))
sns.heatmap(cosine_sim_matrix.iloc[:top_n, :top_n], cmap='coolwarm', annot=False)
plt.title(f'Heatmap Cosine Similarity Antar Top {top_n} Tempat Wisata')
plt.show()

# 3. Distribusi Jumlah Rating per User & per Item
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
rating_counts_user = rating_df['User_Id'].value_counts()
sns.histplot(rating_counts_user, bins=30, kde=True)
plt.title('Distribusi Jumlah Rating per User')
plt.xlabel('Jumlah Rating')
plt.ylabel('Jumlah User')

plt.subplot(1,2,2)
rating_counts_place = rating_df['Place_Id'].value_counts()
sns.histplot(rating_counts_place, bins=30, kde=True, color='orange')
plt.title('Distribusi Jumlah Rating per Tempat Wisata')
plt.xlabel('Jumlah Rating')
plt.ylabel('Jumlah Tempat')

plt.tight_layout()
plt.show()

# 4. Scatter Plot Rating Prediksi vs Aktual
plt.figure(figsize=(8,6))
plt.scatter(y_true, y_pred, alpha=0.5)
plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
plt.xlabel('Rating Aktual')
plt.ylabel('Rating Prediksi')
plt.title('Scatter Plot Rating Prediksi vs Aktual')
plt.show()

# 5. Bar Plot Kategori Wisata Favorit dari Rekomendasi User
if not isinstance(hasil
