import streamlit as st
from PIL import Image


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
    ## Problem Statement:
    Competitive programming is highly popular and immensely valuable for students during placement preparation, 
    as it aids in comprehending Data Structures and Algorithms (DSA). As part of an Information Retrieval package, 
    integrating competitive programming can be achieved by recommending problems to users. Various methods exist to accomplish this. 
    Ranking and displaying problems on the website are fundamental aspects of the project.
    
    ## Approach:
    The project is divided into two main problems: ranking and recommendation. 
    In the ranking aspect, we experiment with several well-known PageRank algorithms and compare their results 
    with the default PageRank algorithm. In the recommendation part, we will explore different techniques, ranging 
    from basic word embeddings to more complex embeddings, to convert text into embeddings for measuring similarity.

    1. Web scraping : Prepare the leetcode dataset from the leetcode website
    2. Ranking : GooglePage Pagerank , NetworkX page rank , Random Walk
    3. Recommendation : TF IDF and Transformers approach 
    """
)


image = Image.open('Images\leetcode.png')
st.image(image,use_column_width=True)



image = Image.open('Images\Search-Engines-and-Page-Ranking.png')
st.image(image,use_column_width=True)