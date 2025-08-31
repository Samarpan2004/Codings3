import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import numpy as np
import networkx as nx
import string

stop_words = set(stopwords.words('english'))

def sentence_similarity(s1, s2):
    w1 = [w.lower() for w in word_tokenize(s1) if w.isalpha() and w.lower() not in stop_words]
    w2 = [w.lower() for w in word_tokenize(s2) if w.isalpha() and w.lower() not in stop_words]
    if not w1 or not w2:
        return 0.0
    all_words = list(set(w1 + w2))
    v1 = [w1.count(w) for w in all_words]
    v2 = [w2.count(w) for w in all_words]
    v1 = np.array(v1); v2 = np.array(v2)
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2))
    return float(np.dot(v1, v2) / denom) if denom != 0 else 0.0

def build_similarity_matrix(sentences):
    n = len(sentences)
    sim_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                sim_mat[i][j] = sentence_similarity(sentences[i], sentences[j])
    return sim_mat

def summarize(text, n_sentences=3):
    sentences = sent_tokenize(text)
    if len(sentences) <= n_sentences:
        return text
    sim_mat = build_similarity_matrix(sentences)
    nx_graph = nx.from_numpy_array(sim_mat)
    scores = nx.pagerank(nx_graph)
    ranked_sentences = sorted(((scores[i], s, i) for i, s in enumerate(sentences)), reverse=True)
    selected_idx = sorted([t[2] for t in ranked_sentences[:n_sentences]])
    summary = " ".join([sentences[i] for i in selected_idx])
    return summary

if __name__ == "__main__":
    sample = """Artificial Intelligence is one of the most transformative technologies of the 21st century.
    It is impacting industries from healthcare to finance, changing how we live and work.
    However, AI also brings challenges such as ethical concerns, job displacement, and bias in algorithms.
    Balancing innovation with responsibility is key for a sustainable AI future."""
    print("Summary:\n", summarize(sample, 2))
