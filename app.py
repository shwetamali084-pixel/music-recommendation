import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# 🎵 Page setup
st.set_page_config(page_title="🎵 Music Recommender System", layout="wide")

# 🌈 Custom background CSS
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://wallpapercave.com/wp/wp9156382.jpg");
    background-size: cover;
}
[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}
.song-card {
    border: 2px solid black;
    border-radius: 8px;
    padding: 10px;
    margin: 10px 0;
    background-color: #f9f9f9;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

st.title("🎵 Music Recommender System")

# 📂 Load data
org_df = pd.read_csv("music_data.csv")   
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

# 🎶 UI: Song selection
songs = org_df['name'].tolist()
selected_song = st.selectbox("Select a song", songs)

if st.button("Recommend"):
    try:
        song_index = org_df[org_df['name'] == selected_song].index[0]
        st.subheader(f"🎵 Recommendations for **{selected_song}**")
        recs = recommend(song_index, 5)

        # Show each recommendation in separate block
        for i in range(len(recs)):
            song_name = recs.loc[i, 'name']
            artists = recs.loc[i, 'artists']
            popularity = recs.loc[i, 'popularity']

            st.markdown(
                f"""
                <div class="song-card">
                    <h4>{i+1}. {song_name}</h4>
                    <p><b>Artists:</b> {artists}</p>
                    <p><b>Popularity:</b> {popularity}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
    except IndexError:
        st.error("❌ Song not found. Please check the spelling.")

# 📌 Sidebar
st.sidebar.title("🎶 Navigation")

# 1. Home
st.sidebar.header("🏠 Home")
st.sidebar.write("Hello Sneha 👋")

# 2. Explore Categories
st.sidebar.header("📂 Explore Categories")

with st.sidebar.expander("90's Songs"):
    if st.sidebar.button("Open 90's Songs"):
        st.sidebar.image("images/90s.jpg", caption="90's Songs", use_container_width=True)

with st.sidebar.expander("60's Songs"):
    if st.sidebar.button("Open 60's Songs"):
        st.sidebar.image("images/60s.jpg", caption="60's Songs", use_container_width=True)

with st.sidebar.expander("Rap Hits"):
    if st.sidebar.button("Open Rap Songs"):
        st.sidebar.image("images/Rap.jpg", caption="Rap Songs", use_container_width=True)

with st.sidebar.expander("Marathi Songs"):
    if st.sidebar.button("Open Marathi Songs"):
        st.sidebar.image("images/Marathi.jpg", caption="Marathi Songs", use_container_width=True)

# 3. SRK Hits
st.sidebar.header("🎬 SRK Hits")
if st.sidebar.button("Open SRK Songs"):
    st.sidebar.image("images/SRK.jpg", caption="SRK Songs", use_container_width=True)

# 4. Vijay Thalapathy Hits
st.sidebar.header("🔥 Vijay Thalapathy Hits")
if st.sidebar.button("Open Vijay Songs"):
    st.sidebar.image("images/Vijay.jpg", caption="Vijay Thalapathy Songs", use_container_width=True)

# 5. Shreya Ghoshal Hits
st.sidebar.header("🎤 Shreya Ghoshal Hits")
if st.sidebar.button("Open Shreya Songs"):
    st.sidebar.image("images/Shreya.jpg", caption="Shreya Songs", use_container_width=True)

# 6. Sonu Nigam Hits
st.sidebar.header("🎶 Sonu Nigam Hits")
if st.sidebar.button("Open Sonu Songs"):
    st.sidebar.image("images/Sonu.jpg", caption="Sonu Songs", use_container_width=True)

# Footer
st.sidebar.markdown("---")
st.sidebar.write("✨ Created by Sneha Ghodke ✨")
