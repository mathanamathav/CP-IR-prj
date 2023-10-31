import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


# Function to recommend similar questions
def recommend_similar_questions(title, num_recommendations=5):
    df = pd.read_csv("Data/leetcode_questions.csv")

    similar_questions = df[["Similar Questions ID", "Question Title", "Similar Questions Text"]]

    # Handle missing text data by replacing NaN with an empty string
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

