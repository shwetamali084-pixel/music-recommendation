import os
import json
import hashlib
from datetime import datetime
import zipfile
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# =========================================================
# 📁 BASE DIRECTORY & FILES
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USERS_FILE = os.path.join(BASE_DIR, "users.json")
ZIP_FILE = os.path.join(BASE_DIR, "Data.zip")

# =========================================================
# 🎵 PAGE SETUP
# =========================================================
st.set_page_config(
    page_title="🎵 Music Recommender System",
    layout="wide"
)

st.title("🎵 Music Recommender System")

# =========================================================
# 👤 USER ACCOUNT & LOGIN FUNCTIONS
# =========================================================
def load_users():
    if not os.path.exists(USERS_FILE):
        return {}
    try:
        with open(USERS_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)
            return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}

def save_users(users):
    try:
        with open(USERS_FILE, "w", encoding="utf-8") as file:
            json.dump(users, file, indent=4, ensure_ascii=False)
        return True
    except OSError:
        return False

def hash_password(password):
    return hashlib.sha256(password.encode("utf-8")).hexdigest()

if "current_user" not in st.session_state:
    st.session_state.current_user = None

if "users_data" not in st.session_state:
    st.session_state.users_data = load_users()

def create_user(username, password, confirm_password):
    username = username.strip()
    if not username:
        return False, "Please enter a username."
    if not password:
        return False, "Please enter a password."
    if len(password) < 6:
        return False, "Password must contain at least 6 characters."
    if password != confirm_password:
        return False, "Passwords do not match."
    if username in st.session_state.users_data:
        return False, "Username already exists. Please login."

    st.session_state.users_data[username] = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "password": hash_password(password),
        "listening_history": []
    }

    if not save_users(st.session_state.users_data):
        del st.session_state.users_data[username]
        return False, "Could not save account."

    st.session_state.current_user = username
    return True, f"Account '{username}' created successfully!"

def login_user(username, password):
    username = username.strip()
    if not username or not password:
        return False, "Please enter username and password."
    if username not in st.session_state.users_data:
        return False, "Username not found."

    user_data = st.session_state.users_data[username]
    if user_data.get("password") != hash_password(password):
        return False, "Incorrect password."

    st.session_state.current_user = username
    return True, f"Welcome back, {username}!"

def add_to_listening_history(song_name, artists="", popularity=""):
    username = st.session_state.current_user
    if not username or username not in st.session_state.users_data:
        return False

    history = st.session_state.users_data[username].setdefault("listening_history", [])
    history.insert(0, {
        "song": str(song_name),
        "artist": str(artists),
        "popularity": str(popularity),
        "listened_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    st.session_state.users_data[username]["listening_history"] = history[:50]
    return save_users(st.session_state.users_data)

def logout_user():
    st.session_state.current_user = None

# =========================================================
# 👤 SIDEBAR USER SECTION
# =========================================================
st.sidebar.header("👤 User Account")

if st.session_state.current_user is None:
    with st.sidebar.expander("🔐 Login", expanded=True):
        login_username = st.text_input("Username", key="login_username")
        login_password = st.text_input("Password", type="password", key="login_password")
        if st.button("🔑 Login", key="login_btn", use_container_width=True):
            success, message = login_user(login_username, login_password)
            if success:
                st.success(message)
                st.rerun()
            else:
                st.error(message)

    with st.sidebar.expander("➕ Create Account"):
        new_username = st.text_input("New username", key="new_username")
        new_password = st.text_input("New password", type="password", key="new_password")
        confirm_password = st.text_input("Confirm password", type="password", key="confirm_password")
        if st.button("Create Account", key="create_user_btn", use_container_width=True):
            success, message = create_user(new_username, new_password, confirm_password)
            if success:
                st.success(message)
                st.rerun()
            else:
                st.error(message)
else:
    st.sidebar.success(f"👋 Logged in as: {st.session_state.current_user}")
    if st.sidebar.button("🚪 Logout", key="logout_btn", use_container_width=True):
        logout_user()
        st.rerun()

    with st.sidebar.expander("🎧 My Listening History"):
        history = st.session_state.users_data.get(st.session_state.current_user, {}).get("listening_history", [])
        if history:
            for i, item in enumerate(history, start=1):
                st.markdown(f"**{i}. {item.get('song')}**\nArtist: {item.get('artist')}")
                st.markdown("---")
            if st.button("🗑️ Clear History"):
                st.session_state.users_data[st.session_state.current_user]["listening_history"] = []
                save_users(st.session_state.users_data)
                st.rerun()
        else:
            st.write("No listening history yet.")

# =========================================================
# 📂 LOAD DATA FROM ZIP
# =========================================================
try:
    with zipfile.ZipFile(ZIP_FILE) as z:
        with z.open("Data.csv") as f:
            org_df = pd.read_csv(f)
except FileNotFoundError:
    st.error("❌ Data.zip file not found in your project directory!")
    st.stop()

# Normalize column names to avoid case/space mismatches
org_df.columns = org_df.columns.str.strip().str.lower()

# Automatically identify the song name column
song_column = None
for col in ["name", "track_name", "title", "song"]:
    if col in org_df.columns:
        song_column = col
        break

if not song_column:
    st.error(f"❌ Could not find a song title column! Available columns: {org_df.columns.tolist()}")
    st.stop()

# Required feature columns check
required_features = [
    "valence", "danceability", "energy", "tempo",
    "acousticness", "instrumentalness", "speechiness",
    "popularity", "explicit"
]

missing_cols = [c for c in required_features if c not in org_df.columns]
if missing_cols:
    st.error(f"❌ Missing required columns in dataset: {missing_cols}")
    st.stop()

df = org_df[required_features].copy()
for col in required_features:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna()
org_df = org_df.loc[df.index].reset_index(drop=True)
df = df.reset_index(drop=True)

# =========================================================
# 🔧 SCALE & MODEL SETUP
# =========================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

nn = NearestNeighbors(n_neighbors=10, metric='cosine')
nn.fit(X_scaled)

def recommend(song_index, n=5):
    distances, indices = nn.kneighbors([X_scaled[song_index]])
    
    # Ensure artist column exists safely
    artist_col = "artists" if "artists" in org_df.columns else org_df.columns[1]
    pop_col = "popularity" if "popularity" in org_df.columns else org_df.columns[0]
    
    recs = org_df.iloc[indices[0][1:n+1]][[song_column, artist_col, pop_col]]
    recs.columns = ["Song Name", "Artists", "Popularity"]
    return recs.reset_index(drop=True)

# =========================================================
# 🎶 SONG SELECTION & RECOMMENDATION UI
# =========================================================
songs = org_df[song_column].dropna().tolist()
selected_song = st.selectbox("🎵 Select a song", songs)

if st.button("Recommend"):
    try:
        song_index = org_df[org_df[song_column] == selected_song].index[0]
        st.subheader(f"🎵 Recommendations for **{selected_song}**")
        
        # Option to mark as listened if logged in
        if st.session_state.current_user:
            row = org_df.iloc[song_index]
            artist_val = row.get("artists", "Unknown")
            pop_val = row.get("popularity", "")
            if st.button("🎧 Mark Selected Song as Listened"):
                add_to_listening_history(selected_song, artist_val, pop_val)
                st.success("Added to listening history!")
                st.rerun()

        rec_df = recommend(song_index, 5)
        st.table(rec_df)

    except IndexError:
        st.error("❌ Song not found. Please check the spelling.")

# =========================================================
# 🖼️ SIDEBAR CATEGORY IMAGES
# =========================================================
st.sidebar.header("🎶 Explore Categories")

with st.sidebar.expander("90's Songs"):
    if st.button("Open 90's Image", key="btn_90s"):
        st.sidebar.image("https://images.unsplash.com/photo-1511671782779-c97d3d27a1d4?w=500", caption="90's Hits", use_container_width=True)

with st.sidebar.expander("60's Songs"):
    if st.button("Open 60's Image", key="btn_60s"):
        st.sidebar.image("https://images.unsplash.com/photo-1465847899084-d164df4dedc6?w=500", caption="60's Classics", use_container_width=True)

with st.sidebar.expander("Rap Hits"):
    if st.button("Open Rap Image", key="btn_rap"):
        st.sidebar.image("https://images.unsplash.com/photo-1508700115892-45ecd05ae2ad?w=500", caption="Rap Hits", use_container_width=True)

with st.sidebar.expander("Marathi Songs"):
    if st.button("Open Marathi Image", key="btn_marathi"):
        st.sidebar.image("https://images.unsplash.com/photo-1514525253161-7a46d19cd819?w=500", caption="Marathi Songs", use_container_width=True)

st.balloons()
