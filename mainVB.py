import os
import math
from collections import Counter

DOC_FOLDER = "documents"

def preprocess(text):
    return text.lower().split()

# Load Documents
documents = {}

for file in os.listdir(DOC_FOLDER):
    if file.endswith(".txt"):
        with open(os.path.join(DOC_FOLDER, file), "r", encoding="utf-8") as f:
            documents[file] = preprocess(f.read())

N = len(documents)

# Document Frequency
df = {}

for words in documents.values():
    for term in set(words):
        df[term] = df.get(term, 0) + 1

# Inverse Document Frequency
idf = {}

for term, freq in df.items():
    idf[term] = math.log(N / freq)

# TF-IDF for Documents
doc_vectors = {}

for doc, words in documents.items():

    tf = Counter(words)
    vector = {}

    for term, freq in tf.items():
        vector[term] = freq * idf.get(term, 0)

    doc_vectors[doc] = vector

# Query Input
query = input("Enter Query: ").lower().split()

query_tf = Counter(query)
query_vector = {}

for term, freq in query_tf.items():
    query_vector[term] = freq * idf.get(term, 0)

print("\n==============================")
print("QUERY TERMS")
print("==============================")
print(query)

print("\n==============================")
print("QUERY TF")
print("==============================")

for term, freq in query_tf.items():
    print(f"{term} : {freq}")

print("\n==============================")
print("IDF VALUES")
print("==============================")

for term in query:
    if term in idf:
        print(f"{term} : {idf[term]:.4f}")
    else:
        print(f"{term} : Not Found")

print("\n==============================")
print("QUERY TF-IDF WEIGHTS")
print("==============================")

for term, weight in query_vector.items():
    print(f"{term} : {weight:.4f}")

# Cosine Similarity
def cosine_similarity(v1, v2):

    common = set(v1.keys()) & set(v2.keys())

    dot = sum(v1[t] * v2[t] for t in common)

    mag1 = math.sqrt(sum(v**2 for v in v1.values()))
    mag2 = math.sqrt(sum(v**2 for v in v2.values()))

    if mag1 == 0 or mag2 == 0:
        return 0

    return dot / (mag1 * mag2)

scores = {}

print("\n==============================")
print("DOCUMENT SIMILARITY SCORES")
print("==============================")

for doc, vector in doc_vectors.items():

    score = cosine_similarity(query_vector, vector)
    scores[doc] = score

    print(f"{doc} : {score:.4f}")

# Ranking
ranked = sorted(
    scores.items(),
    key=lambda x: x[1],
    reverse=True
)

print("\n==============================")
print("FINAL DOCUMENT RANKING")
print("==============================")

for i, (doc, score) in enumerate(ranked, 1):
    print(f"{i}. {doc} -> {score:.4f}")