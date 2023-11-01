import streamlit as st
from utils import (
    random_walk,
    preprocess_convert_graph,
    plot_rank_graph,
    adj_rank,
    eign_rank,
    rank_to_text,
)
import pandas as pd
import networkx as nx

st.set_page_config(page_title="Ranking", page_icon="1️⃣", layout="wide")

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
    # Ranking CP questions

    In an Information Retrieval (IR) project managing a coding question dataset, the aim is to suggest relevant and high-quality coding problems to users. 
    Relevance is determined by alignment with user-selected topics, while quality is assessed based on user success rates. 
    This approach is reminiscent of PageRank, Google's original search engine algorithm, which improved web search results by considering content relevance 
    and link quality. The IR project leverages data-driven insights to enhance the user's learning experience by recommending problems aligned with their 
    interests and ranking them based on their success rates.
    """
)

G = preprocess_convert_graph()
pagerank = nx.pagerank(G)
st.plotly_chart(
    plot_rank_graph(G, "Networkx Package", pagerank), use_container_width=True
)

st.markdown(
    """
    ## Random Walk

    In the context of our project, we employ a random walk with a parameter called alpha to ensure efficient navigation through the coding question dataset. 
    This random walk involves selecting a node and choosing one of its out-links, while also allowing for occasional random jumps to other nodes. 
    The alpha parameter controls the likelihood of following links, with (1-alpha) governing the probability of making random jumps. 
    This approach helps us avoid getting stuck in loops and efficiently explores the dataset, ultimately providing us with a counter that normalizes node 
    visit frequencies, which can be used to approximate PageRank-like results for ranking coding problems.
    """
)

ranks_rw = random_walk(G)

st.plotly_chart(plot_rank_graph(G, "Random Walk", ranks_rw), use_container_width=True)

s1 = pd.Series(pagerank.values())
s2 = pd.Series(ranks_rw)

df = pd.DataFrame(dict(PageRank=s1, RandomWalk=s2))
df["Diff"] = df["RandomWalk"] - df["PageRank"]
df = df * 100
st.dataframe(df, use_container_width=True)

st.markdown(
    """
    ## Adjacency Matrix

    The random walk implementation of PageRank is conceptually simple, 
    but not very efficient to compute. An alternative is to use a matrix to represent the links 
    from each node to every other node, and compute the eigenvectors of that matrix.
    """
)

ranks_am = adj_rank(G)

s1 = pd.Series(pagerank.values())
s2 = pd.Series(ranks_am)

df = pd.DataFrame(dict(PageRank=s1, AdjMatrix=s2))
df["Diff"] = df["AdjMatrix"] - df["PageRank"]
df = df * 100
st.dataframe(df, use_container_width=True)


st.markdown(
    """
    ## Eigenvectors

    If you start with almost any vector and multiply by a matrix repeatedly,as we did in the previous section, the result will converge to the eigenvector of the matrix that corresponds to the largest eigenvalue.
    In fact, repeated multiplication is one of the algorithms used to compute eigenvalues: it is called power iteration.
    Instead of using an iterative method, we can also compute eigenvalues directly, 
    which is what the Numpy function eig does. Here are the eigenvalues and eigenvectors of the Google matrix.
    """
)

ranks_ev = eign_rank(G)

s1 = pd.Series(pagerank.values())
s2 = pd.Series(ranks_ev)

df = pd.DataFrame(dict(PageRank=s1, Eigenvector=s2))
df["Diff"] = df["Eigenvector"] - df["PageRank"]
df = df * 100
st.dataframe(df, use_container_width=True)

st.markdown(
    """
    ## Ranked Top 10 questions
    """
)

st.dataframe(rank_to_text(pagerank), use_container_width=True)
