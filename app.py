import os
import json
import ast
import hashlib
from datetime import datetime

import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors


# =========================================================
# BASE DIRECTORY
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

IMAGE_DIR = BASE_DIR

USERS_FILE = os.path.join(
    BASE_DIR,
    "users.json"
)

DATA_FILE = os.path.join(
    BASE_DIR,
    "music_data.csv"
)


# =========================================================
# STREAMLIT CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="🎵 Music Recommender System",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# USER MANAGEMENT
# =========================================================
def load_users():

    if not os.path.exists(USERS_FILE):
        return {}

    try:
        with open(
            USERS_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

            if isinstance(data, dict):
                return data

    except Exception:
        pass

    return {}


def save_users(users):

    try:
        with open(
            USERS_FILE,
            "w",
            encoding="utf-8"
        ) as file:

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


# =========================================================
# SESSION STATE
# =========================================================
if "current_user" not in st.session_state:
    st.session_state.current_user = None

if "users_data" not in st.session_state:
    st.session_state.users_data = load_users()


# =========================================================
# CREATE USER
# =========================================================
def create_user(
    username,
    password,
    confirm_password
):

    username = username.strip()

    if not username:
        return False, "Please enter a username."

    if not password:
        return False, "Please enter a password."

    if len(password) < 6:
        return (
            False,
            "Password must contain at least 6 characters."
        )

    if password != confirm_password:
        return False, "Passwords do not match."

    if username in st.session_state.users_data:
        return (
            False,
            "Username already exists."
        )

    st.session_state.users_data[username] = {
        "created_at":
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),

        "password":
            hash_password(password),

        "listening_history": []
    }

    if not save_users(
        st.session_state.users_data
    ):

        del st.session_state.users_data[username]

        return (
            False,
            "Could not save account."
        )

    st.session_state.current_user = username

    return (
        True,
        f"Account '{username}' created successfully!"
    )


# =========================================================
# LOGIN
# =========================================================
def login_user(
    username,
    password
):

    username = username.strip()

    if not username or not password:
        return (
            False,
            "Please enter username and password."
        )

    if username not in st.session_state.users_data:
        return (
            False,
            "Username not found."
        )

    user_data = (
        st.session_state.users_data[username]
    )

    saved_password = user_data.get(
        "password",
        ""
    )

    if saved_password != hash_password(password):
        return (
            False,
            "Incorrect password."
        )

    st.session_state.current_user = username

    return (
        True,
        f"Welcome back, {username}!"
    )


# =========================================================
# LISTENING HISTORY
# =========================================================
def add_to_listening_history(
    song_name,
    artist="",
    popularity=""
):

    username = st.session_state.current_user

    if (
        not username
        or username not in st.session_state.users_data
    ):
        return False

    history = (
        st.session_state
        .users_data[username]
        .setdefault(
            "listening_history",
            []
        )
    )

    history = [
        item
        for item in history
        if str(
            item.get("song", "")
        ) != str(song_name)
    ]

    history.insert(
        0,
        {
            "song": str(song_name),
            "artist": str(artist),
            "popularity": str(popularity),
            "listened_at":
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
        }
    )

    st.session_state.users_data[
        username
    ]["listening_history"] = history[:50]

    return save_users(
        st.session_state.users_data
    )


# =========================================================
# SIDEBAR USER ACCOUNT
# =========================================================
st.sidebar.header("👤 User Account")


# =========================================================
# LOGIN / CREATE ACCOUNT
# =========================================================
if st.session_state.current_user is None:

    # -----------------------------------------------------
    # LOGIN
    # -----------------------------------------------------
    with st.sidebar.expander(
        "🔐 Login",
        expanded=True
    ):

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

    # -----------------------------------------------------
    # CREATE ACCOUNT
    # -----------------------------------------------------
    with st.sidebar.expander(
        "➕ Create Account"
    ):

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


# =========================================================
# LOGGED-IN USER
# =========================================================
else:

    st.sidebar.success(
        f"👋 Logged in as: "
        f"**{st.session_state.current_user}**"
    )

    # -----------------------------------------------------
    # LOGOUT
    # -----------------------------------------------------
    if st.sidebar.button(
        "🚪 Logout",
        key="logout_btn",
        width="stretch"
    ):

        st.session_state.current_user = None
        st.rerun()

    # -----------------------------------------------------
    # LISTENING HISTORY
    # -----------------------------------------------------
    with st.sidebar.expander(
        "🎧 Listening History"
    ):

        history = (
            st.session_state
            .users_data
            .get(
                st.session_state.current_user,
                {}
            )
            .get(
                "listening_history",
                []
            )
        )

        if history:

            for i, item in enumerate(
                history,
                start=1
            ):

                song_name = item.get(
                    "song",
                    "Unknown Song"
                )

                artist_name = item.get(
                    "artist",
                    "Unknown Artist"
                )

                st.markdown(
                    f"**{i}. {song_name}**  \n"
                    f"*{artist_name}*"
                )

            if st.button(
                "🗑️ Clear History",
                key="clear_history_btn"
            ):

                st.session_state.users_data[
                    st.session_state.current_user
                ]["listening_history"] = []

                save_users(
                    st.session_state.users_data
                )

                st.rerun()

        else:

            st.write(
                "No history recorded yet."
            )


# =========================================================
# ARTIST PARSER
# =========================================================
def parse_artist_string(
    artist_value
):

    if pd.isna(artist_value):
        return "Unknown Artist"

    text_value = str(
        artist_value
    ).strip()

    if (
        text_value.startswith("[")
        and text_value.endswith("]")
    ):

        try:

            parsed = ast.literal_eval(
                text_value
            )

            if isinstance(
                parsed,
                list
            ):

                artists = [
                    str(x).strip()
                    for x in parsed
                    if str(x).strip()
                ]

                if artists:
                    return ", ".join(
                        artists
                    )

        except Exception:
            pass

    cleaned = (
        text_value
        .strip("[]'\"")
        .strip()
    )

    if cleaned:
        return cleaned

    return "Unknown Artist"


# =========================================================
# LOAD AND PREPROCESS DATASET
# =========================================================
@st.cache_data(show_spinner=False)
def load_and_preprocess_data(filepath):

    if not os.path.isfile(filepath):

        return (
            None,
            [],
            "Dataset file was not found."
        )

    try:

        df = pd.read_csv(
            filepath,
            on_bad_lines="skip",
            low_memory=False,
            encoding="utf-8-sig"
        )

    except UnicodeDecodeError:

        try:

            df = pd.read_csv(
                filepath,
                on_bad_lines="skip",
                low_memory=False,
                encoding="latin1"
            )

        except Exception as error:

            return (
                None,
                [],
                f"CSV encoding error: {error}"
            )

    except Exception as error:

        return (
            None,
            [],
            f"CSV loading error: {error}"
        )

    if df is None or df.empty:

        return (
            None,
            [],
            "The CSV file is empty."
        )

    # -----------------------------------------------------
    # NORMALIZE COLUMN NAMES
    # -----------------------------------------------------
    df.columns = [
        str(column)
        .strip()
        .lower()
        .replace("\ufeff", "")
        for column in df.columns
    ]

    # -----------------------------------------------------
    # TITLE COLUMN
    # -----------------------------------------------------
    title_candidates = [
        "name",
        "title",
        "song_name",
        "track_name",
        "track"
    ]

    title_col = next(
        (
            column
            for column in title_candidates
            if column in df.columns
        ),
        None
    )

    if title_col:

        df["display_title"] = (
            df[title_col]
            .fillna("Unknown Song")
            .astype(str)
            .str.strip()
        )

        df.loc[
            df["display_title"].isin(
                [
                    "",
                    "nan",
                    "None"
                ]
            ),
            "display_title"
        ] = "Unknown Song"

    else:

        df["display_title"] = [
            f"Track {i + 1}"
            for i in range(len(df))
        ]

    # -----------------------------------------------------
    # ARTIST COLUMN
    # -----------------------------------------------------
    artist_candidates = [
        "artists",
        "artist",
        "artist_name",
        "artist_names"
    ]

    artist_col = next(
        (
            column
            for column in artist_candidates
            if column in df.columns
        ),
        None
    )

    if artist_col:

        df["display_artist"] = (
            df[artist_col]
            .apply(parse_artist_string)
        )

    else:

        df["display_artist"] = (
            "Unknown Artist"
        )

    # -----------------------------------------------------
    # POPULARITY
    # -----------------------------------------------------
    if "popularity" not in df.columns:

        df["popularity"] = "N/A"

    else:

        df["popularity"] = (
            df["popularity"]
            .fillna("N/A")
        )

    # -----------------------------------------------------
    # AUDIO FEATURES
    # -----------------------------------------------------
    feature_cols = [
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
        column
        for column in feature_cols
        if column in df.columns
    ]

    # -----------------------------------------------------
    # IF NO AUDIO FEATURES
    # -----------------------------------------------------
    if not available_features:

        return (
            pd.DataFrame(),
            [],
            "No audio feature columns found."
        )

    # -----------------------------------------------------
    # CONVERT FEATURES TO NUMERIC
    # -----------------------------------------------------
    for column in available_features:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # -----------------------------------------------------
    # CLEAN INVALID FEATURE VALUES
    # -----------------------------------------------------
    df = df.dropna(
        subset=available_features
    ).reset_index(drop=True)

    if df.empty:

        return (
            pd.DataFrame(),
            available_features,
            "No rows contain valid audio feature data."
        )

    return (
        df,
        available_features,
        ""
    )


# =========================================================
# RECOMMENDATION ENGINE
# =========================================================
@st.cache_resource(show_spinner=False)
def build_recommendation_engine(
    df,
    available_features
):

    if (
        df is None
        or df.empty
        or not available_features
        or len(df) < 2
    ):

        return (
            None,
            None,
            None
        )

    try:

        scaler = MinMaxScaler()

        X_scaled = scaler.fit_transform(
            df[available_features]
        )

    except Exception:

        return (
            None,
            None,
            None
        )

    try:

        n_neighbors = len(df)

        nn_model = NearestNeighbors(
            n_neighbors=n_neighbors,
            metric="cosine",
            algorithm="brute"
        )

        nn_model.fit(X_scaled)

    except Exception:

        return (
            scaler,
            None,
            X_scaled
        )

    return (
        scaler,
        nn_model,
        X_scaled
    )


# =========================================================
# LOAD DATA
# =========================================================
org_df, features, data_error = (
    load_and_preprocess_data(
        DATA_FILE
    )
)


# =========================================================
# DATASET ERROR
# =========================================================
if (
    org_df is None
    or org_df.empty
):

    st.error(
        "❌ Dataset file missing, empty, "
        "or could not be loaded."
    )

    st.info(
        f"""
Expected dataset:

`music_data.csv`

Location:

`{BASE_DIR}`
        """
    )

    if data_error:

        st.warning(
            f"Details: {data_error}"
        )

    st.stop()


# =========================================================
# FEATURE ERROR
# =========================================================
if not features:

    st.error(
        "❌ No usable audio-feature columns "
        "were found in the CSV."
    )

    st.stop()


# =========================================================
# BUILD MODEL
# =========================================================
scaler, nn_model, X_scaled = (
    build_recommendation_engine(
        org_df,
        features
    )
)


# =========================================================
# MAIN PAGE
# =========================================================
st.title(
    "🎵 Music Recommender System"
)

st.caption(
    f"Dataset: music_data.csv | "
    f"Songs loaded: {len(org_df):,}"
)


# =========================================================
# SONG DROPDOWN
# =========================================================
dropdown_options = org_df.apply(
    lambda row:
        f"{row['display_title']} — "
        f"{row['display_artist']}",
    axis=1
).tolist()


if not dropdown_options:

    st.error(
        "❌ No songs found in dataset."
    )

    st.stop()


selected_option = st.selectbox(
    "🎵 Select a song from dataset:",
    dropdown_options
)


# =========================================================
# RECOMMEND EXACTLY 5 SONGS
# =========================================================
if st.button(
    "🚀 Recommend 5 Songs",
    type="primary"
):

    # -----------------------------------------------------
    # SELECTED SONG INDEX
    # -----------------------------------------------------
    selected_index = (
        dropdown_options.index(
            selected_option
        )
    )

    selected_row = org_df.iloc[
        selected_index
    ]

    selected_title = str(
        selected_row["display_title"]
    )

    selected_artist = str(
        selected_row["display_artist"]
    )

    selected_popularity = str(
        selected_row["popularity"]
    )

    # -----------------------------------------------------
    # SAVE SELECTED SONG
    # -----------------------------------------------------
    if st.session_state.current_user:

        add_to_listening_history(
            selected_title,
            selected_artist,
            selected_popularity
        )

    # -----------------------------------------------------
    # RECOMMENDATION LIST
    # -----------------------------------------------------
    recommended_indices = []

    # =====================================================
    # METHOD 1 - SIMILAR
