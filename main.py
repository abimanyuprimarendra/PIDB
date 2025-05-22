import streamlit as st
import pandas as pd
import numpy as np
import time
from sklearn.metrics.pairwise import cosine_similarity
from heapq import nlargest
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# ---------- LOAD DATASET ----------
@st.cache_data(show_spinner=True)
def load_data():
    csv_url_tour = "https://drive.google.com/uc?id=1toXFdx4bIbDevyPSmEdbs2gG3PR9iYI-"
    csv_url_rating = "https://drive.google.com/uc?id=1NUbzdY_ZNVI2Gc9avZaTvQNT6gp5tc4y"

    tour_df = pd.read_csv(csv_url_tour, dtype=str)
    rating_df = pd.read_csv(csv_url_rating, dtype=str)
    
    tour_df = tour_df.apply(lambda col: col.str.strip() if col.dtype == 'object' else col)
    rating_df = rating_df.apply(lambda col: col.str.strip() if col.dtype == 'object' else col)
    
    rating_df['User_Id'] = rating_df['User_Id'].astype(float).astype(int).astype(str)
    rating_df['Place_Id'] = rating_df['Place_Id'].astype(float).astype(int).astype(str)
    tour_df['Place_Id'] = tour_df['Place_Id'].astype(float).astype(int).astype(str)
    
    rating_df['Place_Ratings'] = pd.to_numeric(rating_df['Place_Ratings'], errors='coerce')
    rating_df.dropna(subset=['User_Id', 'Place_Id', 'Place_Ratings'], inplace=True)
    rating_df.drop_duplicates(inplace=True)
    tour_df.dropna(subset=['Place_Name'], inplace=True)
    
    tour_df['Latitude'] = pd.to_numeric(tour_df['Latitude'], errors='coerce')
    tour_df['Longitude'] = pd.to_numeric(tour_df['Longitude'], errors='coerce')
    
    return tour_df, rating_df

# ---------- PREPARASI DATA & METODE PREDIKSI ----------
@st.cache_data(show_spinner=False)
def prepare_data(tour_df, rating_df):
    user_item_matrix = rating_df.pivot_table(index='User_Id', columns='Place_Id', values='Place_Ratings')
    user_item_matrix = user_item_matrix.fillna(0)
    item_similarity = user_item_matrix.corr(method='pearson')
    item_similarity = item_similarity.fillna(0)
    mean_ratings_dict = user_item_matrix.replace(0, np.NaN).mean().to_dict()
    
    return user_item_matrix, item_similarity, mean_ratings_dict, tour_df, rating_df

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
    mean_neighbors = user_item_matrix[list(rated_neighbors.keys())].replace(0, np.NaN).mean()

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

    from tqdm import tqdm
    from concurrent.futures import ThreadPoolExecutor
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

def recommend_places(selected_place_id, top_k_neighbors, tour_df, k=5):
    neighbors = top_k_neighbors.get(selected_place_id, {})
    neighbors_sorted = sorted(neighbors.items(), key=lambda x: x[1], reverse=True)
    top_recommendations = neighbors_sorted[:k]
    place_ids = [pid for pid, sim in top_recommendations]
    recommended_places = tour_df[tour_df['Place_Id'].isin(place_ids)]
    return recommended_places

# ========== STREAMLIT UI ==========

st.title("Sistem Rekomendasi Tempat Wisata Yogyakarta")

with st.spinner('Loading data...'):
    tour_df, rating_df = load_data()
    user_item_matrix, item_similarity_filled, mean_ratings_dict, tour_df, rating_df = prepare_data(tour_df, rating_df)

k = st.sidebar.slider("Pilih nilai k (jumlah tetangga):", 2, 10, 6)

top_k_neighbors = precompute_top_k_neighbors(item_similarity_filled, k=k)

# Hitung evaluasi model sekali dan cache
@st.cache_data
def cached_evaluate():
    return evaluate_model(user_item_matrix, rating_df, mean_ratings_dict, top_k_neighbors)
mae, rmse, waktu = cached_evaluate()

st.header("Rekomendasi Tempat Wisata")

place_options = tour_df[['Place_Id', 'Place_Name']].drop_duplicates().sort_values('Place_Name')
place_name_list = place_options['Place_Name'].tolist()
place_id_list = place_options['Place_Id'].tolist()

selected_place_name = st.selectbox("Pilih Tempat Wisata:", place_name_list)
selected_place_id = place_options[place_options['Place_Name'] == selected_place_name]['Place_Id'].values[0]

recommended_places = recommend_places(selected_place_id, top_k_neighbors, tour_df, k=5)

st.subheader(f"Rekomendasi tempat mirip dengan {selected_place_name}:")
for idx, row in recommended_places.iterrows():
    st.markdown(f"**{row['Place_Name']}**")

# Tampilkan hasil evaluasi model di bawah rekomendasi
st.markdown("---")
st.subheader("Evaluasi Model")
st.write(f"MAE: {mae:.4f}")
st.write(f"RMSE: {rmse:.4f}")
st.write(f"Waktu Prediksi: {waktu:.2f} ms")

# Tampilkan map dengan lokasi wisata yang direkomendasikan dan tempat yang dipilih
map_df = recommended_places[['Latitude', 'Longitude', 'Place_Name']].copy()
selected_place_coords = tour_df[tour_df['Place_Id'] == selected_place_id][['Latitude', 'Longitude']]

if not selected_place_coords.empty:
    selected_place_coords = selected_place_coords.iloc[0]
    # Tambah tempat terpilih dengan marker khusus
    map_df = pd.concat([
        map_df,
        pd.DataFrame({
            'Latitude': [selected_place_coords['Latitude']],
            'Longitude': [selected_place_coords['Longitude']],
            'Place_Name': [selected_place_name]
        })
    ])

map_df = map_df.dropna(subset=['Latitude', 'Longitude'])

if not map_df.empty:
    st.subheader("Peta Lokasi Tempat Wisata")
    st.map(map_df.rename(columns={'Latitude': 'lat', 'Longitude': 'lon'}))
else:
    st.write("Data koordinat lokasi tidak tersedia untuk peta.")
