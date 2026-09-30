import os
import json
import ast
import hashlib
import base64
import re
from datetime import datetime
import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors

# =========================================================
# 📁 BASE DIRECTORY SETUP
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, "images")
USERS_FILE = os.path.join(BASE_DIR, "users.json")

# Primary dataset
# cleaned_data.csv is already uploaded to the GitHub repository.
DATA_FILE = os.path.join(BASE_DIR, "cleaned_data.csv")


# =========================================================
# 🎵 STREAMLIT PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="🎵 Music Recommender System",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# 🌈 CUSTOM BACKGROUND & CSS
# =========================================================
background_image_path = os.path.join(IMAGE_DIR, "main.jpg")

if os.path.exists(background_image_path):
    with open(background_image_path, "rb") as image_file:
        encoded_bg = base64.b64encode(image_file.read()).decode()
    
    page_bg = f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background-image: url("data:image/jpeg;base64,{encoded_bg}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    [data-testid="stHeader"] {{
        background: rgba(0, 0, 0, 0);
    }}
    .song-card {{
        border: 1px solid #1DB954;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        background-color: rgba(25, 20, 20, 0.88);
        color: #FFFFFF;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.3);
    }}
    .song-card h4 {{
        margin: 0 0 5px 0;
        color: #1DB954;
    }}
    </style>
    """
    st.markdown(page_bg, unsafe_allow_html=True)


# =========================================================
# 👤 USER ACCOUNT MANAGEMENT & LISTENING HISTORY
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
        return False, "Username already exists. Choose another."

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
    saved_password = user_data.get("password")

    if saved_password != hash_password(password):
        return False, "Incorrect password."

    st.session_state.current_user = username
    return True, f"Welcome back, {username}!"


def add_to_listening_history(song_name, artists="", popularity=""):
    username = st.session_state.current_user
    if not username or username not in st.session_state.users_data:
        return False

    history = st.session_state.users_data[username].setdefault("listening_history", [])
    
    if not history or history[0].get("song") != str(song_name):
        history.insert(0, {
            "song": str(song_name),
            "artist": str(artists),
            "popularity": str(popularity),
            "listened_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        st.session_state.users_data[username]["listening_history"] = history[:50]
        return save_users(st.session_state.users_data)
    return True


# =========================================================
# 👤 SIDEBAR USER CONTROLS
# =========================================================
st.sidebar.header("👤 User Account")

if st.session_state.current_user is None:
    with st.sidebar.expander("🔐 Login", expanded=True):
        login_username = st.text_input("Username", key="login_username")
        login_password = st.text_input("Password", type="password", key="login_password")

        if st.button("🔑 Login", key="login_btn", width='stretch'):
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

        if st.button("Create Account", key="create_user_btn", width='stretch'):
            success, message = create_user(new_username, new_password, confirm_password)
            if success:
                st.success(message)
                st.rerun()
            else:
                st.error(message)
else:
    st.sidebar.success(f"👋 Logged in as: **{st.session_state.current_user}**")
    if st.sidebar.button("🚪 Logout", key="logout_btn", width='stretch'):
        st.session_state.current_user = None
        st.rerun()

    with st.sidebar.expander("🎧 Listening History"):
        history = st.session_state.users_data.get(st.session_state.current_user, {}).get("listening_history", [])
        if history:
            for i, item in enumerate(history, start=1):
                st.markdown(f"**{i}. {item.get('song')}**  \n*{item.get('artist')}*")
            if st.button("🗑️ Clear History", key="clear_history_btn"):
                st.session_state.users_data[st.session_state.current_user]["listening_history"] = []
                save_users(st.session_state.users_data)
                st.rerun()
        else:
            st.write("No history recorded yet.")


# =========================================================
# 🎵 DATA PIPELINE & RECOMMENDATION ENGINE
# =========================================================
def parse_artist_string(artist_val):
    """Parses artist strings like ['Artist 1', 'Artist 2'] into 'Artist 1, Artist 2'."""
    if pd.isna(artist_val):
        return "Unknown Artist"
    s = str(artist_val).strip()
    if s.startswith("[") and s.endswith("]"):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                return ", ".join([str(x) for x in parsed])
        except (ValueError, SyntaxError):
            pass
    return s.strip("[]'\"")


@st.cache_data
def load_and_preprocess_data(filepath):
    """Loads CSV and extracts song titles and parsed artist names from dataset."""
    if not os.path.exists(filepath):
        return None, []

    df = pd.read_csv(filepath, on_bad_lines='skip')
    df.columns = [str(c).strip().lower() for c in df.columns]

    # Target specific dataset columns 'name' and 'artists'
    title_col = 'name' if 'name' in df.columns else next((c for c in ['title', 'song_name', 'track_name'] if c in df.columns), None)
    artist_col = 'artists' if 'artists' in df.columns else next((c for c in ['artist', 'artist_name'] if c in df.columns), None)

    df['display_title'] = df[title_col].astype(str).str.strip() if title_col else [f"Track {i+1}" for i in range(len(df))]
    df['display_artist'] = df[artist_col].apply(parse_artist_string) if artist_col else "Unknown Artist"

    if 'popularity' not in df.columns:
        df['popularity'] = "N/A"

    # Audio feature extraction for recommendations
    feature_cols = ["valence", "acousticness", "danceability", "energy", "instrumentalness", "liveness", "loudness", "speechiness", "tempo"]
    available_features = [c for c in feature_cols if c in df.columns]
    
    for c in available_features:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # If the dataset has no usable audio features, return an empty result
    # instead of causing a later TypeError/KeyError.
    if not available_features:
        return pd.DataFrame(), []

    df = df.dropna(subset=available_features).reset_index(drop=True)
    return df, available_features


@st.cache_resource
def build_recommendation_engine(df, available_features):
    """Normalizes audio features and trains NearestNeighbors model."""
    if df is None or df.empty or not available_features:
        return None, None, None

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(df[available_features])

    n_neighbors = min(11, len(X_scaled))
    if n_neighbors < 2:
        return scaler, None, X_scaled

    nn = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric="cosine",
        algorithm="brute"
    )
    nn.fit(X_scaled)
    return scaler, nn, X_scaled


# Load dataset and prepare model
org_df, features = load_and_preprocess_data(DATA_FILE)

if org_df is None or org_df.empty:
    st.error(
        "❌ Dataset file missing, empty, or could not be loaded. "
        f"Please ensure `cleaned_data.csv` is located in:\n`{BASE_DIR}`"
    )
    st.stop()

if not features:
    st.error(
        "❌ No usable audio-feature columns were found in the CSV. "
        "Required columns include at least one of: "
        "`valence`, `acousticness`, `danceability`, `energy`, "
        "`instrumentalness`, `liveness`, `loudness`, `speechiness`, `tempo`."
    )
    st.stop()

scaler, nn_model, X_scaled = build_recommendation_engine(org_df, features)

if nn_model is None or X_scaled is None or len(org_df) < 2:
    st.error("❌ At least 2 songs with valid audio-feature data are required for recommendations.")
    st.stop()


# =========================================================
# 🎶 MAIN RECOMMENDATION UI
# =========================================================
st.title("🎵 Music Recommender System")

# Build song + artist selection list directly from dataset
dropdown_options = org_df.apply(lambda r: f"{r['display_title']} — {r['display_artist']}", axis=1).tolist()
selected_option = st.selectbox("🎵 Select a song from dataset:", dropdown_options)

if st.button("🚀 Recommend 5 Songs", type="primary"):
    selected_index = dropdown_options.index(selected_option)
    selected_row = org_df.iloc[selected_index]
    
    sel_title = selected_row["display_title"]
    sel_artist = selected_row["display_artist"]
    sel_pop = selected_row["popularity"]

    st.subheader(f"🎵 Top 5 Recommended Songs for **{sel_title}** by *{sel_artist}*")

    if st.session_state.current_user:
        add_to_listening_history(sel_title, sel_artist, sel_pop)

    # Find closest 5 neighbors (excluding the selected song itself)
    distances, indices = nn_model.kneighbors([X_scaled[selected_index]])
    rec_indices = indices[0][1:6]

    for rank, idx in enumerate(rec_indices, 1):
        rec_row = org_df.iloc[idx]
        rec_title = rec_row["display_title"]
        rec_artist = rec_row["display_artist"]
        rec_pop = rec_row["popularity"]

        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown(
                f"""
                <div class="song-card">
                    <h4>{rank}. {rec_title}</h4>
                    <p><b>Artist:</b> {rec_artist}</p>
                    <p><b>Popularity:</b> {rec_pop}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        with col2:
            st.write("")
            if st.session_state.current_user:
                if st.button("🎧 Mark Listened", key=f"btn_rec_{idx}_{rank}"):
                    add_to_listening_history(rec_title, rec_artist, rec_pop)
                    st.toast(f"Added '{rec_title}' to history!")


# =========================================================
# 📌 SIDEBAR CATEGORY EXPLORER
# =========================================================
st.sidebar.markdown("---")
st.sidebar.title("🎶 Navigation")
st.sidebar.write("Hello Shweta 👋")

st.sidebar.header("📂 Explore Categories")

def show_category_image(button_label, filename, caption, key):
    with st.sidebar.expander(button_label):
        if st.button(f"Open {button_label}", key=key):
            image_path = os.path.join(IMAGE_DIR, filename)
            if os.path.exists(image_path):
                st.sidebar.image(image_path, caption=caption, width='stretch')
            else:
                st.sidebar.error(f"❌ File `{filename}` not found in images folder!")

show_category_image("90's Songs", "90s.jpeg", "90's Songs", "btn_90s")
show_category_image("60's Songs", "60s.jpg.jpeg", "60's Songs", "btn_60s")
show_category_image("Rap Hits", "Rap.jpg.jpeg", "Rap Songs", "btn_rap")
show_category_image("Marathi Songs", "Marathi.jpg.jpeg", "Marathi Songs", "btn_marathi")

st.sidebar.header("🎬 Artist Hits")

artist_gallery = [
    ("Open SRK Songs", "SRK.jpg.jpeg", "SRK Songs"),
    ("Open Vijay Songs", "Vijay.jpg.jpeg", "Vijay Songs"),
    ("Open Shreya Songs", "Shreya.jpg.jpeg", "Shreya Songs"),
    ("Open Sonu Songs", "Sonu.jpg.jpeg", "Sonu Songs")
]

for label, filename, caption in artist_gallery:
    if st.sidebar.button(label, key=f"btn_artist_{filename}"):
        path = os.path.join(IMAGE_DIR, filename)
        if os.path.exists(path):
            st.sidebar.image(path, caption=caption, width='stretch')
        else:
            st.sidebar.error(f"❌ File `{filename}` not found!")

st.sidebar.markdown("---")
st.sidebar.write("✨ Created by Shweta Mali ✨")
