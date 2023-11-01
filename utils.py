import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from collections import Counter
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import networkx as nx

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

    p = np.full(N, 1/N)
    alpha = 0.85
    GM = alpha * M + (1 - alpha) * p
    x = np.full(N, 1/N)

    for i in range(10):
        x = GM.T @ x

    ranks_am = x / x.sum()
    return ranks_am

def preprocess_convert_graph():
    df = pd.read_csv("Data/leetcode_questions.csv")
    graph_df = df[['Question ID' , 'Similar Questions ID' , 'Question Title']]
    graph_df.dropna( inplace = True )

    graph_df['Similar Questions ID'] = graph_df['Similar Questions ID'].str.split(',')
    graph_df = graph_df.explode('Similar Questions ID')
    graph_df = graph_df.rename(columns={'Similar Questions ID': 'Similar Question ID'})
    graph_df = graph_df.reset_index(drop=True)
    graph_df['Similar Question ID'] = graph_df['Similar Question ID'].astype('int')
    G = nx.from_pandas_edgelist(graph_df, "Question ID", "Similar Question ID")
    return G

def plot_rank_graph(G , title , ranks):
    fig = make_subplots(rows=1, cols=1)
    pos = nx.spring_layout(G)

    # Create node positions for Plotly
    node_x = [pos[k][0] for k in G.nodes]
    node_y = [pos[k][1] for k in G.nodes]

    # Create a list of labels with PageRank values
    labels = [f'Node {k}<br>PageRank: {ranks[k]:.3f}' for k in G.nodes]

    # Create a scatter plot for nodes
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        marker=dict(size=[v * 5000 for v in ranks.values()], showscale=True, colorscale='YlGnBu' ),
        text=labels,
        hoverinfo='text'
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
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    # Add traces to the figure
    fig.add_trace(edge_trace)
    fig.add_trace(node_trace)

    # Update layout
    fig.update_layout(
        showlegend=False,
        hovermode='closest',
        title_text='Interactive {} PageRank Graph'.format(title),
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

def recommend_similar_questions(title, num_recommendations=5):
    df = pd.read_csv("Data/leetcode_questions.csv")

    similar_questions = df[["Similar Questions ID", "Question Title", "Similar Questions Text"]]

    similar_questions["Similar Questions Text"].fillna("", inplace=True)

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(similar_questions["Similar Questions Text"])

    cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)
    title_to_index = {title: index for index, title in enumerate(df["Question Title"])}

    question_index = title_to_index.get(title)

    if question_index is None:
        return []

    sim_scores = list(enumerate(cosine_similarities[question_index]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    similar_question_indices = [
        index for index, _ in sim_scores[1 : num_recommendations + 1]
    ]

    return df["Question Title"].iloc[similar_question_indices]

