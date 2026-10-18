import re
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def parse_cisi(file_path, is_query=False):
    records = []
    current = None
    field = None

    with open(file_path, "r", encoding="latin-1") as f:
        for line in f:
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

            elif current is not None and field:
                current[field] += " " + line.strip()

    if current is not None:
        records.append(current)

    return pd.DataFrame(records)


documents = parse_cisi("CISI.ALL")
queries = parse_cisi("CISI.QRY", is_query=True)

print("Total documents:", len(documents))
print("Total queries:", len(queries))
documents.head()



def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


documents["clean_text"] = (
    documents["title"] + " " + documents["text"]
).apply(preprocess)

queries["clean_text"] = queries["text"].apply(preprocess)

print(documents[["id", "clean_text"]].head())


vectorizer = TfidfVectorizer(
    stop_words="english",
    max_df=0.95,
    min_df=1
)

tfidf_matrix = vectorizer.fit_transform(
    documents["clean_text"]
)

print("Document matrix shape:", tfidf_matrix.shape)
print("Vocabulary size:", len(vectorizer.vocabulary_))

def search_documents(query, top_k=10):
    query = preprocess(query)
    query_vector = vectorizer.transform([query])

    scores = cosine_similarity(
        query_vector, tfidf_matrix
    ).flatten()

    top_indices = scores.argsort()[::-1][:top_k]

    results = documents.iloc[top_indices][
        ["id", "title"]
    ].copy()

    results["similarity"] = scores[top_indices]
    results = results[results["similarity"] > 0]

    return results.reset_index(drop=True)


search_results = search_documents(
    "information retrieval systems",
    top_k=10
)

print(search_results)


# Load query-document relevance pairs
relevance = pd.read_csv(
    "CISI.REL",
    sep=r"\s+",
    header=None,
    usecols=[0, 1],
    names=["query_id", "doc_id"]
)

relevant_docs = {}

for _, row in relevance.iterrows():
    relevant_docs.setdefault(
        int(row["query_id"]), set()
    ).add(int(row["doc_id"]))


def evaluate_system(k=10):
    precision_scores = []
    recall_scores = []
    average_precisions = []

    for _, query_row in queries.iterrows():
        qid = int(query_row["id"])
        actual = relevant_docs.get(qid, set())

        if not actual:
            continue

        results = search_documents(
            query_row["text"], top_k=k
        )
        retrieved = results["id"].astype(int).tolist()

        hits = [
            1 if doc_id in actual else 0
            for doc_id in retrieved
        ]

        relevant_count = sum(hits)

        precision_scores.append(relevant_count / k)
        recall_scores.append(relevant_count / len(actual))

        precisions_at_hits = [
            sum(hits[:i + 1]) / (i + 1)
            for i, hit in enumerate(hits) if hit
        ]

        average_precisions.append(
            sum(precisions_at_hits) / len(actual)
        )

    print("Precision@10:", round(
        sum(precision_scores) / len(precision_scores), 4
    ))
    print("Recall@10:", round(
        sum(recall_scores) / len(recall_scores), 4
    ))
    print("MAP:", round(
        sum(average_precisions) / len(average_precisions), 4
    ))


evaluate_system()


plt.figure(figsize=(10, 5))

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

