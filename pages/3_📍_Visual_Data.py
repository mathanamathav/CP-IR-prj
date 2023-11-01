import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from community import community_louvain
from utils import (
    preprocess_convert_graph,
)
import pandas as pd
import networkx as nx
import plotly.express as px

st.set_page_config(page_title="Visualization of Dataset", page_icon="📊", layout="wide")
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
    # Visualizing the dataset
    Utilize various plotting techniques to gain a better understanding of the dataset and make informed decisions.
    """
)


G ,df = preprocess_convert_graph()

st.dataframe(df, use_container_width=True)

plot_options = {"node_size": 10, "with_labels": False, "width": 0.15}
pos = nx.spring_layout(G, iterations=15, seed=1721)
fig, ax = plt.subplots(figsize=(15, 9))
ax.axis("off")
nx.draw_networkx(G, pos=pos, ax=ax, **plot_options)
plt.title("Leetcode NetworkX Graph")
st.pyplot(plt.gcf())


G_undirected = G.to_undirected()
partition = community_louvain.best_partition(G_undirected)
community_groups = {}
for node, community_id in partition.items():
    if community_id not in community_groups:
        community_groups[community_id] = []
    community_groups[community_id].append(node)
community_graph = nx.Graph()
for community_id, nodes in community_groups.items():
    community_graph.add_nodes_from(nodes, community=community_id)
for node, community_id in partition.items():
    for neighbor in G_undirected.neighbors(node):
        if partition[neighbor] == community_id:
            community_graph.add_edge(node, neighbor)

pos = nx.spring_layout(community_graph, seed=42)
colors = [partition[node] for node in community_graph.nodes()]
nx.draw(community_graph, pos, node_color=colors, with_labels=False, cmap=plt.cm.get_cmap('viridis'))
st.pyplot(plt.gcf())


count_data = {}
for val in df["Difficulty Level"].tolist():
    count_data[val] = 1 + count_data.get(val , 0)
    
data = {
    'Category': count_data.keys(),
    'Count': count_data.values()
}
cdf = pd.DataFrame(data)
fig = px.bar(cdf, x='Category', y='Count', title='Bar Chart for Categorical Column')
st.plotly_chart(fig , use_container_width=True)

fig = px.line(df, x='Question ID', y=['Success Rate', 'Likes', 'Dislikes'], title='Performance Over Time')
fig.update_xaxes(rangeslider_visible=True)
st.plotly_chart(fig , use_container_width=True)


data = {
    'topic': df["Topic Tagged text"]
}
sdf = pd.DataFrame(data)
topic_counts = sdf['topic'].str.split(',').explode().value_counts().reset_index()
topic_counts.columns = ['Topic', 'Count']

fig = px.bar(topic_counts, x='Topic', y='Count', title='Tagged Topics Count')
st.plotly_chart(fig , use_container_width=True)

topics = [str(topic) for topic in data['topic']]
text = ','.join(topics)

wordcloud = WordCloud(width=1000, height=600, background_color='white').generate(text)
st.image(wordcloud.to_array() ,use_column_width=True)