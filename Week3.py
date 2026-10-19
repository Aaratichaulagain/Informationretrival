
import os
import re
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =====================================================
# STEP 1: CREATE PROJECT FOLDERS
# =====================================================

project_folder = "Document_Similarity_Assignment"
documents_folder = os.path.join(project_folder, "documents")
results_folder = os.path.join(project_folder, "results")

os.makedirs(documents_folder, exist_ok=True)
os.makedirs(results_folder, exist_ok=True)

# =====================================================
# STEP 2: ONLINE DOCUMENT SOURCES
# Topic: Artificial Intelligence and Climate Change
# =====================================================

sources = [
    {
        "title": "Aligning AI and Climate Governance",
        "url": "https://www.un.org/en/climatechange/aligning-ai-and-climate-governance"
    },
    {
        "title": "Artificial Intelligence - United Nations",
        "url": "https://www.un.org/en/global-issues/artificial-intelligence"
    },
    {
        "title": "AI for Climate Action",
        "url": "https://unfccc.int/news/ai-for-climate-action-technology-mechanism-supports-transformational-climate-solutions"
    },
    {
        "title": "UN Climate Change",
        "url": "https://unfccc.int/"
    },
    {
        "title": "UNESCO Artificial Intelligence",
        "url": "https://www.unesco.org/en/artificial-intelligence"
    },
    {
        "title": "International Telecommunication Union AI for Good",
        "url": "https://aiforgood.itu.int/"
    },
    {
        "title": "United Nations Environment Programme",
        "url": "https://www.unep.org/"
    },
    {
        "title": "NASA Climate Change",
        "url": "https://science.nasa.gov/climate-change/"
    },
    {
        "title": "World Meteorological Organization",
        "url": "https://wmo.int/"
    },
    {
        "title": "United Nations Development Programme",
        "url": "https://www.undp.org/"
    }
]

# =====================================================
# STEP 3: DOWNLOAD AND SAVE DOCUMENTS AS TXT FILES
# =====================================================

downloaded = []
failed = []

headers = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/130.0.0.0 Safari/537.36"
    )
}

for index, source in enumerate(sources, start=1):

    print(f"\nProcessing: {source['title']}")

    try:
        response = requests.get(
            source["url"],
            headers=headers,
            timeout=30
        )
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove navigation and other unnecessary elements
        for tag in soup.find_all([
            "script", "style", "nav", "footer",
            "header", "aside", "noscript", "form"
        ]):
            tag.decompose()

        # Prefer article/main content where available
        main = soup.find("article") or soup.find("main") or soup.body

        if main is None:
            raise ValueError("No readable webpage content found.")

        text_content = main.get_text(
            separator=" ",
            strip=True
        )

        # Clean unnecessary spaces
        text_content = re.sub(
            r"\s+",
            " ",
            text_content
        ).strip()

        word_count = len(text_content.split())

        # Reject short pages and likely error pages
        if word_count < 40:
            raise ValueError(
                f"Only {word_count} readable words found."
            )

        filename = f"document_{index:02d}.txt"
        filepath = os.path.join(documents_folder, filename)

        # Save document and source details
        with open(filepath, "w", encoding="utf-8") as file:
            file.write(f"Title: {source['title']}\n")
            file.write(f"Source URL: {source['url']}\n\n")
            file.write(text_content)

        downloaded.append({
            "filename": filename,
            "title": source["title"],
            "url": source["url"],
            "text": text_content,
            "word_count": word_count
        })

        print(f"Saved: {filename}")
        print(f"Word count: {word_count}")

    except Exception as error:
        failed.append({
            "title": source["title"],
            "url": source["url"],
            "error": str(error)
        })

        print(f"Download failed: {error}")

# Save source information
if downloaded:
    source_df = pd.DataFrame([
        {
            "Filename": item["filename"],
            "Title": item["title"],
            "Source URL": item["url"],
            "Word Count": item["word_count"]
        }
        for item in downloaded
    ])

    source_df.to_csv(
        os.path.join(results_folder, "document_sources.csv"),
        index=False
    )

# Save information about failed downloads
if failed:
    pd.DataFrame(failed).to_csv(
        os.path.join(results_folder, "failed_downloads.csv"),
        index=False
    )

print("\nSuccessfully downloaded:", len(downloaded))
print("Failed downloads:", len(failed))

# Stop if fewer than five documents were collected
if len(downloaded) < 5:
    raise RuntimeError(
        "Fewer than 5 documents were collected. "
        "Check your internet connection or replace failed URLs "
        "with accessible public articles, then run again."
    )

# =====================================================
# STEP 4: PREPARE DOCUMENT TEXT
# =====================================================

document_names = [
    item["filename"] for item in downloaded
]

document_texts = [
    item["text"] for item in downloaded
]

# =====================================================
# STEP 5: TF-IDF VECTOR SPACE REPRESENTATION
# =====================================================

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    max_features=5000
)

tfidf_matrix = vectorizer.fit_transform(document_texts)

print("\nTF-IDF vectorization completed.")
print("Number of documents:", tfidf_matrix.shape[0])
print("Vocabulary size:", tfidf_matrix.shape[1])

# Save TF-IDF vectors
tfidf_df = pd.DataFrame(
    tfidf_matrix.toarray(),
    index=document_names,
    columns=vectorizer.get_feature_names_out()
)

tfidf_df.to_csv(
    os.path.join(results_folder, "tfidf_vectors.csv")
)

# =====================================================
# STEP 6: CALCULATE COSINE SIMILARITY
# =====================================================

similarity_matrix = cosine_similarity(tfidf_matrix)

similarity_df = pd.DataFrame(
    similarity_matrix,
    index=document_names,
    columns=document_names
)

similarity_df.to_csv(
    os.path.join(results_folder, "cosine_similarity_matrix.csv")
)

print("\nCosine Similarity Matrix:")
print(similarity_df.round(3).to_string())

# =====================================================
# STEP 7: RANK ALL UNIQUE DOCUMENT PAIRS
# =====================================================

pairs = []

for i in range(len(document_names)):
    for j in range(i + 1, len(document_names)):
        pairs.append({
            "Document 1": document_names[i],
            "Document 2": document_names[j],
            "Similarity Score": similarity_matrix[i, j]
        })

pairs_df = pd.DataFrame(pairs)

pairs_df = pairs_df.sort_values(
    by="Similarity Score",
    ascending=False
).reset_index(drop=True)

pairs_df.to_csv(
    os.path.join(results_folder, "ranked_similarity_pairs.csv"),
    index=False
)

print("\nTop 10 most similar document pairs:")
print(pairs_df.head(10).round(4).to_string(index=False))

print("\nLeast similar document pair:")
print(pairs_df.tail(1).round(4).to_string(index=False))

# =====================================================
# STEP 8: CREATE COSINE SIMILARITY HEATMAP
# =====================================================

plt.figure(figsize=(12, 9))

sns.heatmap(
    similarity_df,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    vmin=0,
    vmax=1
)

plt.title("Cosine Similarity Between Documents")
plt.xlabel("Documents")
plt.ylabel("Documents")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()

plt.savefig(
    os.path.join(
        results_folder,
        "cosine_similarity_heatmap.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# =====================================================
# STEP 9: BAR CHART OF TOP 10 SIMILAR PAIRS
# =====================================================

top_pairs = pairs_df.head(10).copy()

top_pairs["Document Pair"] = (
    top_pairs["Document 1"]
    + " vs "
    + top_pairs["Document 2"]
)

plt.figure(figsize=(12, 7))

sns.barplot(
    data=top_pairs,
    x="Similarity Score",
    y="Document Pair"
)

plt.title("Top 10 Most Similar Document Pairs")
plt.xlabel("Cosine Similarity Score")
plt.ylabel("Document Pair")
plt.xlim(0, 1)
plt.tight_layout()

plt.savefig(
    os.path.join(
        results_folder,
        "top_10_similar_pairs.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# =====================================================
# STEP 10: SAVE SUMMARY AND FINDINGS
# =====================================================

highest = pairs_df.iloc[0]
lowest = pairs_df.iloc[-1]

summary = f"""
DOCUMENT SIMILARITY ANALYSIS
============================

Topic: Artificial Intelligence and Climate Change

Successfully collected documents: {len(downloaded)}
Vocabulary size: {tfidf_matrix.shape[1]}
Unique document pairs: {len(pairs_df)}

Method:
TF-IDF Vector Space Model and Cosine Similarity

Most similar document pair:
{highest['Document 1']} and {highest['Document 2']}

Highest similarity score:
{highest['Similarity Score']:.4f}

Least similar document pair:
{lowest['Document 1']} and {lowest['Document 2']}

Lowest similarity score:
{lowest['Similarity Score']:.4f}

Interpretation:
Higher scores indicate more similar word usage in the
TF-IDF document representations. Lower scores indicate
less overlap in the weighted terms.

These scores measure textual similarity, not factual
agreement or similarity of meaning in every context.
"""

with open(
    os.path.join(results_folder, "summary.txt"),
    "w",
    encoding="utf-8"
) as file:
    file.write(summary)

print(summary)

print("\nAssignment completed successfully.")
print("Project folder:", os.path.abspath(project_folder))
print("Results folder:", os.path.abspath(results_folder))
