import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import util,SentenceTransformer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import networkx as nx
import streamlit as st
import re


def rank_to_text(rank):
    sorted_dict = dict(sorted(rank.items(), key=lambda item: item[1], reverse=True))
    df = pd.read_csv("Data/leetcode_questions.csv")
    ids = list(sorted_dict.keys())[:10]
    result_df = df[df["Question ID"].isin(ids)]
    return result_df

def eign_rank(G):
    M = nx.to_numpy_array(G)
    alpha = 0.85

    N = len(G)
    P = np.ones((N, N)) / N
    M = alpha * M + (1 - alpha) * P

    eigenvalues, eigenvectors = np.linalg.eig(M.T)
    ind = np.argmax(eigenvalues)
    largest_eigenvector = np.real(eigenvectors[:, ind])
    ranks_ev = largest_eigenvector / largest_eigenvector.sum()

    return ranks_ev


def adj_rank(G):
    M = nx.to_numpy_array(G)
    N = len(G)

    p = np.full(N, 1 / N)
    alpha = 0.85
    GM = alpha * M + (1 - alpha) * p
    x = np.full(N, 1 / N)

    for i in range(10):
        x = GM.T @ x

    ranks_am = x / x.sum()
    return ranks_am

@st.cache_data
def preprocess_convert_graph():
    df = pd.read_csv("Data/leetcode_questions.csv")
    graph_df = df[["Question ID", "Similar Questions ID", "Question Title"]]
    graph_df.dropna(inplace=True)

    graph_df["Similar Questions ID"] = graph_df["Similar Questions ID"].str.split(",")
    graph_df = graph_df.explode("Similar Questions ID")
    graph_df = graph_df.rename(columns={"Similar Questions ID": "Similar Question ID"})
    graph_df = graph_df.reset_index(drop=True)
    graph_df["Similar Question ID"] = graph_df["Similar Question ID"].astype("int")
    G = nx.from_pandas_edgelist(graph_df, "Question ID", "Similar Question ID")
    return G,df


def plot_rank_graph(G, title, ranks):
    fig = make_subplots(rows=1, cols=1)
    pos = nx.spring_layout(G)

    # Create node positions for Plotly
    node_x = [pos[k][0] for k in G.nodes]
    node_y = [pos[k][1] for k in G.nodes]

    # Create a list of labels with PageRank values
    labels = [f"Node {k}<br>PageRank: {ranks[k]:.3f}" for k in G.nodes]

    # Create a scatter plot for nodes
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        marker=dict(
            size=[v * 5000 for v in ranks.values()], showscale=True, colorscale="YlGnBu"
        ),
        text=labels,
        hoverinfo="text",
    )

    # Create edges
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    # Create an edge trace
    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.5, color="#888"),
        hoverinfo="none",
        mode="lines",
    )

    # Add traces to the figure
    fig.add_trace(edge_trace)
    fig.add_trace(node_trace)

    # Update layout
    fig.update_layout(
        showlegend=False,
        hovermode="closest",
        title_text="Interactive {} PageRank Graph".format(title),
        height=1000,
    )

    return fig


def flip(p):
    return np.random.random() < p

def random_walk(G, alpha=0.85, iters=1000):
    counter = Counter()
    node = next(iter(G))

    for _ in range(iters):
        if flip(alpha):
            node = np.random.choice(list(G[node]))
        else:
            node = np.random.choice(list(G))

        counter[node] += 1

    total = sum(counter.values())
    for key in counter:
        counter[key] /= total
    return counter


def recommend_similar_questions(df,X, vectorizer, query, num_recommendations=5):
    query_vec = vectorizer.transform([query])
    results = cosine_similarity(X, query_vec)
    res = results.flatten()
    out_arr = np.argsort(res)
    return [df.iloc[i, 3] for i in out_arr[-num_recommendations:] if df.iloc[i, 3].strip() != "" ]

def preprocess_text(text):
    cleaned_text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    cleaned_text = cleaned_text.lower()
    return cleaned_text

@st.cache_resource
def create_df_emb():

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    df = pd.read_csv("Data/leetcode_questions.csv")
    df["Question Text"] = df["Question Text"].fillna("")

    df["Question Text"] = df["Question Text"].apply(preprocess_text)
    question1 = df["Question Text"].tolist()
    questions = [item for item in question1 if item != ""]

    embedding = []
    for question in questions:
        embedding.append(model.encode(question, convert_to_tensor=True))

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(questions)

    return df, embedding, questions, vectorizer , X , model


def recommend_similar_questions_using_transformer(
    embedding, questions, model, query, num_recommendations=10
):
    query = preprocess_text(query)
    query_emb = model.encode(query, convert_to_tensor=True)

    score = []
    for emb in embedding:
        score.append(util.pytorch_cos_sim(emb, query_emb).item())

    similar_question_indices = np.argsort(score)[::-1]

    top_similar_questions = [
        questions[i] for i in similar_question_indices[1 : num_recommendations + 1]
    ]
    return top_similar_questions
