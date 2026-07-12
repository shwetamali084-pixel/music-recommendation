import streamlit as st
import pandas as pd
import zipfile
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# 🎵 Page setup
st.set_page_config(page_title="🎵 Music Recommender System", layout="wide")
st.title("Music Recommender System")

# 📂 Load data
with zipfile.ZipFile("data.zip") as z:
    with z.open("data.csv") as f:
        org_df = pd.read_csv(f)
# org_df = pd.read_csv("Data.csv")   
df = org_df[["valence","danceability","energy","tempo",
             "acousticness","instrumentalness","speechiness",
             "popularity","explicit"]]

# 🔧 Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# 🤝 Nearest Neighbors model
nn = NearestNeighbors(n_neighbors=10, metric='cosine')
nn.fit(X_scaled)

# 🎯 Recommendation function
def recommend(song_index, n=5):
    distances, indices = nn.kneighbors([X_scaled[song_index]])
    recs = org_df.iloc[indices[0][1:n+1]][['name','artists','popularity']]
    recs = recs.reset_index(drop=True)
    return recs

songs = org_df['name'].tolist()
selected_song = st.selectbox("Select a song", songs)

if st.button("Recommend"):
    try:
        song_index = org_df[org_df['name'] == selected_song].index[0]
        st.subheader(f"🎵 Recommendations for **{selected_song}**")
        st.table(recommend(song_index, 5))   # show as neat table
    except IndexError:
        st.error("❌ Song not found. Please check the spelling.")

# 🎶 Sidebar Categories
st.sidebar.header("🎶 Explore Categories")

with st.sidebar.expander("90's Songs"):
    if st.button("Open 90's Image", key="btn_90s"):
        st.image("https://i.pinimg.com/564x/7f/7f/2a/real90s.jpg", use_container_width=True)

with st.sidebar.expander("60's Songs"):
    if st.button("Open 60's Image", key="btn_60s"):
        st.image("https://i.pinimg.com/564x/60/60/3b/real60s.jpg", use_container_width=True)

with st.sidebar.expander("Rap Hits"):
    if st.button("Open Rap Image", key="btn_rap"):
        st.image("https://i.pinimg.com/564x/rap/12/34/realrap.jpg", use_container_width=True)

with st.sidebar.expander("Marathi Songs"):
    if st.button("Open Marathi Image", key="btn_marathi"):
        st.image("https://i.pinimg.com/564x/marathi/56/78/realmarathi.jpg", use_container_width=True)

st.balloons()
