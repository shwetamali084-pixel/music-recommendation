import os
import json
import ast
import hashlib
import base64
from datetime import datetime

import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors


# =========================================================
# BASE DIRECTORY SETUP
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Images are stored directly in GitHub root
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
# STREAMLIT PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="🎵 Music Recommender System",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# BACKGROUND IMAGE
# =========================================================
background_image_path = os.path.join(
    IMAGE_DIR,
    "main.jpg"
)

if os.path.exists(background_image_path):

    try:

        with open(
            background_image_path,
            "rb"
        ) as image_file:

            encoded_bg = base64.b64encode(
                image_file.read()
            ).decode()

        page_bg = f"""
        <style>

        [data-testid="stAppViewContainer"] {{
            background-image:
            url("data:image/jpeg;base64,{encoded_bg}");

            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}

        [data-testid="stHeader"] {{
            background: rgba(0, 0, 0, 0);
        }}

        </style>
        """

        st.markdown(
            page_bg,
            unsafe_allow_html=True
        )

    except Exception:
        pass


# =========================================================
# USER ACCOUNT MANAGEMENT
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

            return {}

    except (
        json.JSONDecodeError,
        OSError
    ):

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

    except OSError:

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

        return (
            False,
            "Please enter a username."
        )

    if not password:

        return (
            False,
            "Please enter a password."
        )

    if len(password) < 6:

        return (
            False,
            "Password must contain at least 6 characters."
        )

    if password != confirm_password:

        return (
            False,
            "Passwords do not match."
        )

    if username in st.session_state.users_data:

        return (
            False,
            "Username already exists. Choose another."
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

        del st.session_state.users_data[
            username
        ]

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
        st.session_state.users_data[
            username
        ]
    )

    saved_password = user_data.get(
        "password"
    )

    if saved_password != hash_password(
        password
    ):

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
    artists="",
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

    # Remove duplicate song
    history = [
        item
        for item in history
        if str(
            item.get(
                "song",
                ""
            )
        ) != str(song_name)
    ]

    history.insert(
        0,
        {
            "song": str(song_name),

            "artist": str(artists),

            "popularity": str(popularity),

            "listened_at":
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
        }
    )

    # Keep latest 50 songs
    st.session_state.users_data[
        username
    ]["listening_history"] = history[:50]

    return save_users(
        st.session_state.users_data
    )


# =========================================================
# SIDEBAR USER ACCOUNT
# =========================================================
st.sidebar.header(
    "👤 User Account"
)


# =========================================================
# LOGIN / CREATE ACCOUNT
# =========================================================
if st.session_state.current_user is None:

    # -------------------------
    # LOGIN
    # -------------------------
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


    # -------------------------
    # CREATE ACCOUNT
    # -------------------------
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


    # -------------------------
    # LOGOUT
    # -------------------------
    if st.sidebar.button(
        "🚪 Logout",
        key="logout_btn",
        width="stretch"
    ):

        st.session_state.current_user = None

        st.rerun()


    # -------------------------
    # LISTENING HISTORY
    # -------------------------
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
    artist_val
):

    if pd.isna(artist_val):

        return "Unknown Artist"

    text_value = str(
        artist_val
    ).strip()


    # Handle list format
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
                ]

                if artists:

                    return ", ".join(
                        artists
                    )

        except (
            ValueError,
            SyntaxError
        ):

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
# LOAD DATASET
# =========================================================
@st.cache_data(
    show_spinner=False
)
def load_and_preprocess_data(
    filepath
):

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


    # Normalize column names
    df.columns = [
        str(column)
        .strip()
        .lower()
        .replace("\ufeff", "")
        for column in df.columns
    ]


    # =====================================================
    # SONG NAME
    # =====================================================
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


    # =====================================================
    # ARTIST
    # =====================================================
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


    # =====================================================
    # DISPLAY TITLE
    # =====================================================
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


    # =====================================================
    # DISPLAY ARTIST
    # =====================================================
    if artist_col:

        df["display_artist"] = (
            df[artist_col]
            .apply(
                parse_artist_string
            )
        )

    else:

        df["display_artist"] = (
            "Unknown Artist"
        )


    # =====================================================
    # POPULARITY
    # =====================================================
    if "popularity" not in df.columns:

        df["popularity"] = "N/A"

    else:

        df["popularity"] = (
            df["popularity"]
            .fillna("N/A")
        )


    # =====================================================
    # AUDIO FEATURES
    # =====================================================
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


    # Convert features to numeric
    for column in available_features:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )


    if not available_features:

        return (
            pd.DataFrame(),
            [],
            "No audio feature columns found."
        )


    # Remove invalid rows
    df = df.dropna(
        subset=available_features
    ).reset_index(
        drop=True
    )


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
@st.cache_resource(
    show_spinner=False
)
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


    scaler = MinMaxScaler()


    try:

        X_scaled = scaler.fit_transform(
            df[available_features]
        )

    except Exception:

        return (
            None,
            None,
            None
        )


    n_neighbors = min(
        11,
        len(X_scaled)
    )


    if n_neighbors < 2:

        return (
            scaler,
            None,
            X_scaled
        )


    nn_model = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric="cosine",
        algorithm="brute"
    )

    nn_model.fit(
        X_scaled
    )


    return (
        scaler,
        nn_model,
        X_scaled
    )


# =========================================================
# LOAD DATASET
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
        📁 Expected dataset location:

        `{DATA_FILE}`

        📄 Required filename:

        `music_data.csv`
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
# BUILD RECOMMENDATION MODEL
# =========================================================
scaler, nn_model, X_scaled = (
    build_recommendation_engine(
        org_df,
        features
    )
)


if (
    nn_model is None
    or X_scaled is None
    or len(org_df) < 2
):

    st.error(
        "❌ At least 2 songs with valid "
        "audio-feature data are required."
    )

    st.stop()


# =========================================================
# MAIN RECOMMENDER UI
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


selected_option = st.selectb
