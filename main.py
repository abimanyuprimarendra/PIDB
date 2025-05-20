import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Fungsi untuk memuat data dari Google Drive
@st.cache_data
def load_data():
    url_tour = "https://drive.google.com/uc?export=download&id=1toXFdx4bIbDevyPSmEdbs2gG3PR9iYI"
    url_rating = "https://drive.google.com/uc?export=download&id=1NUbzdY_ZNVI2Gc9avZaTvQNT6gp5tc4y"

    tour_df = pd.read_csv(url_tour)
    rating_df = pd.read_csv(url_rating)

    tour_df.dropna(inplace=True)
    rating_df.dropna(inplace=True)
    tour_df.drop_duplicates(inplace=True)
    rating_df.drop_duplicates(inplace=True)

    merged_df = pd.merge(rating_df, tour_df, on='Place_Id', how='inner')
    return tour_df, rating_df, merged_df

# Load data
tour_df, rating_df, merged_df = load_data()

# Siapkan data untuk IBCF
df_ibcf = merged_df[['User_Id', 'Place_Name', 'Place_Ratings']]
user_item_matrix = df_ibcf.pivot_table(index='User_Id', columns='Place_Name', values='Place_Ratings')
user_mean = user_item_matrix.mean(axis=1)
user_item_matrix_centered = user_item_matrix.sub(user_mean, axis=0)
item_similarity = user_item_matrix_centered.corr(method='pearson', min_periods=2)

# Fungsi prediksi rating
def predict_rating(user_id, item_name, k=6):
    if item_name not in item_similarity.columns:
        return user_item_matrix.stack().mean()
    if user_id not in user_item_matrix.index:
        return user_item_matrix[item_name].mean()

    user_ratings = user_item_matrix.loc[user_id]
    rated_items = user_ratings.dropna()
    similarities = item_similarity[item_name][rated_items.index].dropna()
    top_k = similarities.sort_values(ascending=False).head(k)

    if top_k.empty:
        return user_ratings.mean()

    top_k_ratings = rated_items[top_k.index]
    numerator = np.dot(top_k.values, top_k_ratings.values)
    denominator = np.sum(np.abs(top_k.values))

    if denominator == 0:
        return user_ratings.mean()
    
    return numerator / denominator

# Evaluasi model
def evaluate_model(k=6):
    preds = []
    for idx, row in df_ibcf.iterrows():
        user = row['User_Id']
        place = row['Place_Name']
        actual = row['Place_Ratings']
        pred = predict_rating(user, place, k)
        if not np.isnan(pred):
            preds.append((actual, pred))
    actual_ratings = [x[0] for x in preds]
    predicted_ratings = [x[1] for x in preds]
    mae = mean_absolute_error(actual_ratings, predicted_ratings)
    rmse = np.sqrt(mean_squared_error(actual_ratings, predicted_ratings))
    return mae, rmse, actual_ratings, predicted_ratings

# Streamlit UI
st.title("🧠 Hybrid Recommender System - Wisata Jogja")

user_list = user_item_matrix.index.tolist()
selected_user = st.selectbox("👤 Pilih User ID:", user_list)

if selected_user:
    st.subheader("📊 Riwayat Rating User")
    user_ratings = user_item_matrix.loc[selected_user].dropna()
    st.table(user_ratings.reset_index().rename(columns={selected_user: 'Rating'}))

    unrated_places = user_item_matrix.loc[selected_user][user_item_matrix.loc[selected_user].isna()].index
    recommendations = []
    for place in unrated_places:
        pred = predict_rating(selected_user, place)
        recommendations.append((place, round(pred, 2)))
    
    recommendations.sort(key=lambda x: x[1], reverse=True)
    top_n = 5
    st.subheader(f"🌟 Rekomendasi Top {top_n} Tempat Wisata untuk User {selected_user}")
    for place, rating in recommendations[:top_n]:
        st.markdown(f"- **{place}** — Prediksi Rating: `{rating}`")

    # Evaluasi dan visualisasi
    mae, rmse, actual_ratings, predicted_ratings = evaluate_model(k=6)
    st.subheader("🧪 Evaluasi Model (k=6)")
    st.write(f"**MAE**: `{mae:.4f}`")
    st.write(f"**RMSE**: `{rmse:.4f}`")

    fig, ax = plt.subplots()
    ax.scatter(actual_ratings, predicted_ratings, alpha=0.4)
    ax.plot([1, 5], [1, 5], 'r--')
    ax.set_xlabel('Rating Asli')
    ax.set_ylabel('Rating Prediksi')
    ax.set_title('📈 Perbandingan Rating Asli vs Prediksi')
    st.pyplot(fig)
