import streamlit as st
from utils import (
    recommend_similar_questions,
    recommend_similar_questions_using_transformer,
    create_df_emb,
)

st.set_page_config(page_title="Recommendation", page_icon="⁉️", layout="wide")
df, embedding, questions, vectorizer, X, model = create_df_emb()

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

st.title("CP similar question Recommender")

query = st.text_input("Enter the Query", "")

if st.button("Search"):
    recommended_questions_normal = recommend_similar_questions(
        df, X, vectorizer, query, num_recommendations=5
    )

    recommended_questions = recommend_similar_questions_using_transformer(
        embedding, questions, model, query, num_recommendations=5
    )

    if recommended_questions_normal:
        st.subheader("Normal Method")
        st.write("Recommended Similar Questions:")
        st.dataframe(recommended_questions_normal, use_container_width=True)
    else:
        st.write("No similar questions found for this topic using normal tokens")

    if recommended_questions:
        st.subheader("Transformer Method")
        st.write("Recommended Similar Questions:")
        st.dataframe(recommended_questions, use_container_width=True)
    else:
        st.write("No similar questions found for this topic.")
