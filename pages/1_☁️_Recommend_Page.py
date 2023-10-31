import streamlit as st
from utils import recommend_similar_questions

st.set_page_config(page_title="Recommendation", page_icon="⁉️", layout="wide")

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

st.title("Coding Question Recommender")

question_title = st.text_input("Enter the topic", "")

if st.button("Recommend Similar Questions"):
    recommended_questions = recommend_similar_questions(question_title, num_recommendations=5)

    if recommended_questions:
        st.write("Recommended Similar Questions:")
        st.table(recommended_questions)
    else:
        st.write("No similar questions found for this topic.")
