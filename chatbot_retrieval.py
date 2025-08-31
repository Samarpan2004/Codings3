import sys
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

class RetrievalChatbot:
    def __init__(self, corpus_path="qa_corpus.tsv"):
        self.questions, self.answers = self._load_corpus(corpus_path)
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,2))
        self.q_vectors = self.vectorizer.fit_transform(self.questions)

    def _load_corpus(self, path):
        qs, ans = [], []
        with open(path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    qs.append(parts[0])
                    ans.append(parts[1])
        return qs, ans

    def respond(self, user_input, top_k=1):
        q = user_input.strip()
        if not q:
            return "Say something, please."
        qv = self.vectorizer.transform([q])
        sims = linear_kernel(qv, self.q_vectors).flatten()
        best_idx = sims.argmax()
        if sims[best_idx] < 0.15:
            return "Sorry, I don't know the answer to that yet."
        return self.answers[best_idx]

def main():
    bot = RetrievalChatbot()
    print("Retrieval Chatbot ready. Type 'exit' to quit.")
    while True:
        try:
            user = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user.lower().strip() in ("exit", "quit"):
            print("Bot: Goodbye!")
            break
        print("Bot:", bot.respond(user))

if __name__ == "__main__":
    main()
