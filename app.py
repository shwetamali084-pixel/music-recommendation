import os
import json
import ast
import hashlib
from datetime import datetime

import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_FILE = os.path.join(BASE_DIR, "music_data.csv")
USERS_FILE = os.path.join(BASE_DIR, "users.json")
IMAGE_DIR = BASE_DIR

st.set_page_config(
    page_title="🎵 Music Recommender System",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================
# SESSION
# =========================

if "current_user" not in st.session_state:
    st.session_state.current_user = None

if "users_data" not in st.session_state:
    st.session_state.users_data = {}


if "recommended_songs" not in st.session_state:
    st.session_state.recommended_songs = []

if "selected_song" not in st.session_state:
    st.session_state.selected_song = None


# =========================
# USER FUNCTIONS
# =========================

def load_users():
    if not os.path.exists(USERS_FILE):
        return {}

    try:
        with open(USERS_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)

        return data if isinstance(data, dict) else {}

    except Exception:
        return {}


def save_users(users):
    try:
        with open(USERS_FILE, "w", encoding="utf-8") as file:
            json.dump(
                users,
                file,
                indent=4,
                ensure_ascii=False
            )
        return True

    except Exception:
        return False


def hash_password(password):
    return hashlib.sha256(
        password.encode("utf-8")
    ).hexdigest()


st.session_state.users_data = load_users()


def create_user(username, password, confirm_password):

    username = username.strip()

    if not username:
        return False, "Please enter username."

    if not password:
        return False, "Please enter password."

    if len(password) < 6:
        return False, "Password must contain at least 6 characters."

    if password != confirm_password:
        return False, "Passwords do not match."

    if username in st.session_state.users_data:
        return False, "Username already exists."

    st.session_state.users_data[username] = {
        "created_at": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "password": hash_password(password),
        "listening_history": []
    }

    if save_users(st.session_state.users_data):
        st.session_state.current_user = username
        return True, "Account created successfully!"

    del st.session_state.users_data[username]

    return False, "Could not save account."


def login_user(username, password):

    username = username.strip()

    if not username or not password:
        return False, "Enter username and password."

    if username not in st.session_state.users_data:
        return False, "Username not found."

    saved_password = (
        st.session_state.users_data[username]
        .get("password", "")
    )

    if saved_password != hash_password(password):
        return False, "Incorrect password."

    st.session_state.current_user = username

    return True, f"Welcome back, {username}!"


def add_to_history(song, artist, popularity):

    username = st.session_state.current_user

    if not username:
        return

    history = (
        st.session_state.users_data[username]
        .setdefault("listening_history", [])
    )

    history = [
        x for x in history
        if str(x.get("song", "")) != str(song)
    ]

    history.insert(
        0,
        {
            "song": str(song),
            "artist": str(artist),
            "popularity": str(popularity),
            "listened_at": datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        }
    )

    st.session_state.users_data[username][
        "listening_history"
    ] = history[:50]

    save_users(st.session_state.users_data)


# =========================
# SIDEBAR ACCOUNT
# =========================

st.sidebar.header("👤 User Account")

if st.session_state.current_user is None:

    with st.sidebar.expander("🔐 Login", expanded=True):

        login_username = st.text_input(
            "Username",
            key="login_username"
        )

        login_password = st.text_input(
            "Password",
            type="password",
            key="login_password"
        )

        if st.button(
            "🔑 Login",
            key="login_btn",
            width="stretch"
        ):

            success, message = login_user(
                login_username,
                login_password
            )

            if success:
                st.success(message)
                st.rerun()
            else:
                st.error(message)

    with st.sidebar.expander("➕ Create Account"):

        new_username = st.text_input(
            "New username",
            key="new_username"
        )

        new_password = st.text_input(
            "New password",
            type="password",
            key="new_password"
        )

        confirm_password = st.text_input(
            "Confirm password",
            type="password",
            key="confirm_password"
        )

        if st.button(
            "Create Account",
            key="create_user_btn",
            width="stretch"
        ):

            success, message = create_user(
                new_username,
                new_password,
                confirm_password
            )

            if success:
                st.success(message)
                st.rerun()
            else:
                st.error(message)

else:

    st.sidebar.success(
        f"👋 Logged in as: "
        f"**{st.session_state.current_user}**"
    )

    if st.sidebar.button(
        "🚪 Logout",
        key="logout_btn",
        width="stretch"
    ):

        st.session_state.current_user = None
        st.session_state.recommended_songs = []
        st.session_state.selected_song = None
        st.rerun()

    with st.sidebar.expander("🎧 Listening History"):

        history = (
            st.session_state.users_data
            .get(st.session_state.current_user, {})
            .get("listening_history", [])
        )

        if history:

            for i, item in enumerate(history, 1):

                st.markdown(
                    f"**{i}. {item.get('song', 'Unknown Song')}**  \n"
                    f"*{item.get('artist', 'Unknown Artist')}*"
                )

            if st.button(
                "🗑️ Clear History",
                key="clear_history"
            ):

                st.session_state.users_data[
                    st.session_state.current_user
                ]["listening_history"] = []

                save_users(st.session_state.users_data)
                st.rerun()

        else:
            st.write("No history recorded yet.")


# =========================
# HELPERS
# =========================

def parse_artist(value):

    if pd.isna(value):
        return "Unknown Artist"

    value = str(value).strip()

    if value.startswith("[") and value.endswith("]"):

        try:
            result = ast.literal_eval(value)

            if isinstance(result, list):

                artists = [
                    str(x).strip()
                    for x in result
                    if str(x).strip()
                ]

                if artists:
                    return ", ".join(artists)

        except Exception:
            pass

    value = value.strip("[]'\"").strip()

    return value if value else "Unknown Artist"


def find_column(df, candidates):

    for column in candidates:

        if column in df.columns:
            return column

    return None


# =========================
# LOAD DATA
# =========================

@st.cache_data(show_spinner=False)
def load_data(filepath):

    if not os.path.isfile(filepath):
        return None, [], "music_data.csv was not found."

    try:

        try:
            df = pd.read_csv(
                filepath,
                encoding="utf-8-sig",
                low_memory=False,
                on_bad_lines="skip"
            )

        except UnicodeDecodeError:

            df = pd.read_csv(
                filepath,
                encoding="latin1",
                low_memory=False,
                on_bad_lines="skip"
            )

    except Exception as error:

        return None, [], str(error)

    if df.empty:
        return None, [], "CSV file is empty."

    df.columns = [
        str(c).strip().lower().replace("\ufeff", "")
        for c in df.columns
    ]

    title_col = find_column(
        df,
        [
            "name",
            "track_name",
            "song_name",
            "title",
            "track"
        ]
    )

    if title_col:

        df["display_title"] = (
            df[title_col]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        df.loc[
            df["display_title"].isin(
                ["", "nan", "none", "None"]
            ),
            "display_title"
        ] = "Unknown Song"

    else:

        df["display_title"] = [
            f"Track {i + 1}"
            for i in range(len(df))
        ]

    artist_col = find_column(
        df,
        [
            "artists",
            "artist",
            "artist_name",
            "artist_names"
        ]
    )

    if artist_col:
        df["display_artist"] = (
            df[artist_col].apply(parse_artist)
        )
    else:
        df["display_artist"] = "Unknown Artist"

    if "popularity" in df.columns:

        df["popularity"] = (
            df["popularity"]
            .fillna("N/A")
        )

    else:

        df["popularity"] = "N/A"

    feature_columns = [
        "valence",
        "acousticness",
        "danceability",
        "energy",
        "instrumentalness",
        "liveness",
        "loudness",
        "speechiness",
        "tempo"
    ]

    available_features = [
        c for c in feature_columns
        if c in df.columns
    ]

    if not available_features:
        return None, [], "No audio features found."

    for column in available_features:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df = df.dropna(
        subset=available_features
    ).reset_index(drop=True)

    if df.empty:
        return None, [], "No valid songs found."

    return df, available_features, ""


# =========================
# RECOMMENDATION MODEL
# =========================

@st.cache_resource(show_spinner=False)
def create_model(df, features):

    if len(df) < 2:
        return None, None

    try:

        scaler = MinMaxScaler()

        values = scaler.fit_transform(
            df[features]
        )

        model = NearestNeighbors(
            n_neighbors=len(df),
            metric="cosine",
            algorithm="brute"
        )

        model.fit(values)

        return scaler, model

    except Exception:
        return None, None


# =========================
# DATA
# =========================

df, features, data_error = load_data(DATA_FILE)

if df is None:

    st.error("❌ Dataset could not be loaded.")

    st.info(
        f"Make sure `music_data.csv` is inside:\n\n{BASE_DIR}"
    )

    if data_error:
        st.warning(data_error)

    st.stop()


if len(df) < 6:

    st.error(
        f"❌ At least 6 songs are required to recommend 5 different songs."
    )

    st.info(
        f"Currently valid songs: {len(df)}"
    )

    st.stop()


scaler, model = create_model(df, features)


# =========================
# MAIN PAGE
# =========================

st.title("🎵 Music Recommender System")

st.caption(
    f"Dataset: music_data.csv | "
    f"Songs available: {len(df):,}"
)


# =========================
# SONG SELECTION
# =========================

options = [
    f"{row['display_title']} — {row['display_artist']}"
    for _, row in df.iterrows()
]

selected_option = st.selectbox(
    "🎵 Select a song:",
    options,
    key="song_selector"
)


# =========================
# RECOMMENDATION FUNCTION
# =========================

def get_recommendations(selected_index):

    recommendations = []

    if model is not None and scaler is not None:

        try:

            scaled_data = scaler.transform(
                df[features]
            )

            distances, indices = model.kneighbors(
                [scaled_data[selected_index]],
                n_neighbors=len(df)
            )

            for index in indices[0]:

                index = int(index)

                if index == selected_index:
                    continue

                if index not in recommendations:
                    recommendations.append(index)

                if len(recommendations) == 5:
                    break

        except Exception:
            recommendations = []

    if len(recommendations) < 5:

        for index in range(len(df)):

            if index == selected_index:
                continue

            if index in recommendations:
                continue

            recommendations.append(index)

            if len(recommendations) == 5:
                break

    return recommendations[:5]


# =========================
# RECOMMEND BUTTON
# =========================

if st.button(
    "🚀 Recommend 5 Songs",
    type="primary",
    width="stretch"
):

    selected_index = options.index(
        selected_option
    )

    selected_song = df.iloc[selected_index]

    st.session_state.selected_song = selected_index

    st.session_state.recommended_songs = (
        get_recommendations(selected_index)
    )

    if st.session_state.current_user:

        add_to_history(
            selected_song["display_title"],
            selected_song["display_artist"],
            selected_song["popularity"]
        )


# =========================
# SELECTED SONG
# =========================

if st.session_state.selected_song is not None:

    selected_index = st.session_state.selected_song

    selected_song = df.iloc[selected_index]

    st.markdown("---")

    st.subheader("🎧 Selected Song")

    st.write(
        f"**{selected_song['display_title']}**"
    )

    st.write(
        f"Artist: {selected_song['display_artist']}"
    )


# =========================
# RECOMMENDED SONGS
# =========================

recommended = st.session_state.recommended_songs

if recommended:

    st.markdown("---")

    st.subheader("🎵 Recommended Songs")

    st.success(
        f"✅ {len(recommended)} songs recommended for you!"
    )

    for position, index in enumerate(
        recommended,
        start=1
    ):

        song = df.iloc[index]

        title = str(
            song["display_title"]
        )

        artist = str(
            song["display_artist"]
        )

        popularity = str(
            song["popularity"]
        )

        st.markdown(
            f"### 🎵 {position}. {title}"
        )

        st.write(
            f"**Artist:** {artist}"
        )

        st.write(
            f"**Popularity:** {popularity}"
        )

        if st.session_state.current_user:

            if st.button(
                f"🎧 Mark Listened",
                key=f"listen_{index}_{position}",
                width="stretch"
            ):

                add_to_history(
                    title,
                    artist,
                    popularity
                )

                st.success(
                    f"'{title}' added to listening history."
                )

        st.markdown("---")


# =========================
# IMAGE DISPLAY
# =========================

def display_project_image(
    button_label,
    filenames,
    caption,
    key
):

    with st.sidebar.expander(button_label):

        if st.button(
            f"Open {button_label}",
            key=key,
            width="stretch"
        ):

            found = None

            for filename in filenames:

                path = os.path.join(
                    IMAGE_DIR,
                    filename
                )

                if os.path.isfile(path):
                    found = path
                    break

            if found:

                st.image(
                    found,
                    caption=caption,
                    width="stretch"
                )

            else:

                st.error(
                    "Image file not found."
                )


# =========================
# SIDEBAR CATEGORIES
# =========================

st.sidebar.markdown("---")

st.sidebar.header("📂 Explore Categories")

display_project_image(
    "90's Songs",
    ["90s.jpeg", "90s.jpg", "90s.png"],
    "90's Songs",
    "category_90s"
)

display_project_image(
    "60's Songs",
    ["60s.jpg", "60s.jpg.jpeg", "60s.jpeg", "60s.png"],
    "60's Songs",
    "category_60s"
)

display_project_image(
    "Rap Hits",
    ["Rap.jpg", "Rap.jpg.jpeg", "Rap.png", "rap.jpg", "rap.png"],
    "Rap Songs",
    "category_rap"
)

display_project_image(
    "Marathi Songs",
    [
        "Marathi.jpg",
        "Marathi.jpg.jpeg",
        "Marathi.jpeg",
        "Marathi.png"
    ],
    "Marathi Songs",
    "category_marathi"
)


# =========================
# ARTIST HITS
# =========================

st.sidebar.header("🎬 Artist Hits")

display_project_image(
    "SRK Songs",
    ["SRK.jpg", "SRK.jpg.jpeg", "SRK.jpeg", "SRK.png"],
    "SRK Songs",
    "artist_srk"
)

display_project_image(
    "Vijay Songs",
    ["Vijay.jpg", "Vijay.jpg.jpeg", "Vijay.jpeg", "Vijay.png"],
    "Vijay Songs",
    "artist_vijay"
)

display_project_image(
    "Shreya Songs",
    ["Shreya.jpg", "Shreya.jpg.jpeg", "Shreya.jpeg", "Shreya.png"],
    "Shreya Songs",
    "artist_shreya"
)

display_project_image(
    "Sonu Songs",
    ["Sonu.jpg", "Sonu.jpg.jpeg", "Sonu.jpeg", "Sonu.png"],
    "Sonu Songs",
    "artist_sonu"
)


st.sidebar.markdown("---")

st.sidebar.write("✨ Created by Shweta Mali ✨")
