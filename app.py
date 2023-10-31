import streamlit as st

st.set_page_config(page_title="About", page_icon="🕸️", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        max-width: 100%;
        padding: 1rem;
    }
    .stButton>button {
        background-color: #008B8B;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    # About The Project
    ----
    An Information Retrieval (IR) project entails a dataset of coding questions, with the IR system tasked with:
    Ranking or suggesting questions to users based on their success rates, utilizing data.
    Recommending problems related to a specific topic.
    """
)
