import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =====================================================
# 1. LOAD THE CISI DATASET
# =====================================================

def parse_cisi(file_path, is_query=False):
    records = []
    current = None
    field = None

    with open(file_path, "r", encoding="latin-1") as f:
        for line in f:

            # Start a new document or query.
            if line.startswith(".I"):
                if current is not None:
                    records.append(current)

                current = {
                    "id": int(line.split()[1]),
                    "title": "",
                    "author": "",
                    "text": ""
                }
                field = None

            elif current is not None and line.startswith(".T"):
                field = "title"

            elif current is not None and line.startswith(".A"):
                field = "author"

            elif current is not None and line.startswith(".W"):
                field = "text"

            elif current is not None and line.startswith(".X"):
                field = None

            # Add text to the currently active field.
            elif current is not None and field:
                current[field] += " " + line.strip()

    if current is not None:
        records.append(current)

    return pd.DataFrame(records)


# Load documents and queries.
documents = parse_cisi("CISI.ALL")
queries = parse_cisi("CISI.QRY", is_query=True)

print("Total documents:", len(documents))
print("Total queries:", len(queries))

# Display a sample of the documents.
print(documents.head())


# =====================================================
# 2. TEXT PREPROCESSING
# =====================================================

def preprocess(text):
    # Convert text to lowercase.
    text = text.lower()

    # Remove punctuation, numbers, and other non-letter characters.
    text = re.sub(r"[^a-z\s]", " ", text)

    # Remove extra spaces.
    text = re.sub(r"\s+", " ", text).strip()

    return text


# Combine document titles and text before cleaning.
documents["clean_text"] = (
    documents["title"] + " " + documents["text"]
).apply(preprocess)

# Clean query text.
queries["clean_text"] = queries["text"].apply(preprocess)

print("\nSample preprocessed documents:")
print(documents[["id", "clean_text"]].head())


# =====================================================
# 3. TF-IDF INDEX CONSTRUCTION
# =====================================================

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_df=0.95,
    min_df=1
)

# Convert documents into numerical TF-IDF vectors.
tfidf_matrix = vectorizer.fit_transform(
    documents["clean_text"]
)

print("\nDocument matrix shape:", tfidf_matrix.shape)
print("Vocabulary size:", len(vectorizer.vocabulary_))


# =====================================================
# 4. DOCUMENT RETRIEVAL USING COSINE SIMILARITY
# =====================================================

def search_documents(query, top_k=10):

    # Preprocess the user's query.
    query = preprocess(query)

    # Convert the query into a TF-IDF vector.
    query_vector = vectorizer.transform([query])

    # Calculate similarity between the query and all documents.
    scores = cosine_similarity(
        query_vector, tfidf_matrix
    ).flatten()

    # Rank documents from highest to lowest similarity.
    top_indices = scores.argsort()[::-1][:top_k]

    # Create a table containing the retrieved documents.
    results = documents.iloc[top_indices][
        ["id", "title"]
    ].copy()

    results["similarity"] = scores[top_indices]

    # Keep documents with a positive similarity score.
    results = results[results["similarity"] > 0]

    return results.reset_index(drop=True)


# Test the search system with an example query.
search_results = search_documents(
    "information retrieval systems",
    top_k=10
)

print("\nSearch results:")
print(search_results.to_string(index=False))


# =====================================================
# 5. LOAD RELEVANCE JUDGMENTS
# =====================================================

# Read query-document relevance pairs from CISI.REL.
relevance = pd.read_csv(
    "CISI.REL",
    sep=r"\s+",
    header=None,
    usecols=[0, 1],
    names=["query_id", "doc_id"]
)

# Ensure document and query IDs are integers.
relevance["query_id"] = relevance["query_id"].astype(int)
relevance["doc_id"] = relevance["doc_id"].astype(int)

# Store relevant document IDs for each query.
relevant_docs = {}

for _, row in relevance.iterrows():
    relevant_docs.setdefault(
        int(row["query_id"]), set()
    ).add(int(row["doc_id"]))

print("\nQueries with relevance judgments:", len(relevant_docs))


# =====================================================
# 6. PRECISION@K
# =====================================================

def precision_at_k(ranked_docs, actual, k=10):

    # Count relevant documents among the top k results.
    hits = sum(
        doc_id in actual for doc_id in ranked_docs[:k]
    )

    # Use k as the denominator.
    return hits / k


# =====================================================
# 7. RECALL@K
# =====================================================

def recall_at_k(ranked_docs, actual, k=10):

    # Avoid division by zero.
    if not actual:
        return 0.0

    # Count relevant documents retrieved in the top k.
    hits = sum(
        doc_id in actual for doc_id in ranked_docs[:k]
    )

    # Calculate the fraction of all relevant documents retrieved.
    return hits / len(actual)


# =====================================================
# 8. AVERAGE PRECISION@K
# =====================================================

def average_precision_at_k(ranked_docs, actual, k=10):

    if not actual:
        return 0.0

    hits = 0
    precision_sum = 0.0

    # Examine documents in ranked order.
    for rank, doc_id in enumerate(ranked_docs[:k], start=1):

        # Update the score when a relevant document is found.
        if doc_id in actual:
            hits += 1
            precision_sum += hits / rank

    # Normalize by the total number of relevant documents.
    return precision_sum / len(actual)


# =====================================================
# 9. nDCG@K
# =====================================================

def ndcg_at_k(ranked_docs, actual, k=10):

    # Calculate relevance of the retrieved documents.
    # 1 means relevant; 0 means not relevant.
    relevance_scores = [
        1 if doc_id in actual else 0
        for doc_id in ranked_docs[:k]
    ]

    # Calculate DCG, giving more weight to higher ranks.
    dcg = sum(
        rel / np.log2(rank + 1)
        for rank, rel in enumerate(
            relevance_scores, start=1
        )
    )

    # Build the ideal ranking with relevant documents first.
    ideal_relevance = (
        [1] * min(len(actual), k)
        + [0] * (k - min(len(actual), k))
    )

    # Calculate the ideal DCG.
    idcg = sum(
        rel / np.log2(rank + 1)
        for rank, rel in enumerate(
            ideal_relevance, start=1
        )
    )

    # Return zero if the ideal DCG is zero.
    if idcg == 0:
        return 0.0

    # Normalize DCG by the ideal DCG.
    return dcg / idcg


# =====================================================
# 10. EVALUATE THE SYSTEM ACROSS QUERIES
# =====================================================

def evaluate_system(k=10):

    precision_scores = []
    recall_scores = []
    average_precision_scores = []
    ndcg_scores = []

    # Evaluate every query that has relevance judgments.
    for _, query_row in queries.iterrows():

        qid = int(query_row["id"])
        actual = relevant_docs.get(qid, set())

        # Skip queries without known relevant documents.
        if not actual:
            continue

        # Retrieve ranked documents for the query.
        results = search_documents(
            query_row["text"], top_k=k
        )

        # Get document IDs in ranked order.
        retrieved = results["id"].astype(int).tolist()

        # Calculate all four evaluation metrics.
        precision_scores.append(
            precision_at_k(retrieved, actual, k)
        )

        recall_scores.append(
            recall_at_k(retrieved, actual, k)
        )

        average_precision_scores.append(
            average_precision_at_k(retrieved, actual, k)
        )

        ndcg_scores.append(
            ndcg_at_k(retrieved, actual, k)
        )

    # Display the average evaluation scores.
    if precision_scores:

        print("\n========== EVALUATION RESULTS ==========")

        print(
            f"Precision@{k}:",
            round(np.mean(precision_scores), 4)
        )

        print(
            f"Recall@{k}:",
            round(np.mean(recall_scores), 4)
        )

        print(
            f"MAP@{k}:",
            round(np.mean(average_precision_scores), 4)
        )

        print(
            f"nDCG@{k}:",
            round(np.mean(ndcg_scores), 4)
        )

        print(
            "Queries evaluated:",
            len(precision_scores)
        )

    else:
        print("No queries with relevance judgments were evaluated.")


# Run the evaluation.
evaluate_system(k=10)


# =====================================================
# 11. VISUALIZE THE SEARCH RESULTS
# =====================================================

plt.figure(figsize=(10, 5))

# Plot the similarity score of each retrieved document.
plt.bar(
    search_results["id"].astype(str),
    search_results["similarity"]
)

plt.xlabel("Document ID")
plt.ylabel("Cosine Similarity")
plt.title("Top Retrieved Documents")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()