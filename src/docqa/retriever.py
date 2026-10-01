from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .chunker import Chunk


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float


class Retriever:
    """Finds the passages most similar to a question using TF-IDF."""

    def __init__(self, chunks):
        self.chunks = list(chunks)
        self._vectorizer = TfidfVectorizer(
            stop_words="english", ngram_range=(1, 2), sublinear_tf=True
        )
        self._matrix = None
        if self.chunks:
            try:
                self._matrix = self._vectorizer.fit_transform(
                    [chunk.text for chunk in self.chunks]
                )
            except ValueError:
                # The documents contain only stop words, so there is nothing to index
                self._matrix = None

    def search(self, question, top_k=3, min_score=0.05):
        """Return up to `top_k` passages, best match first, skipping weak matches."""
        if self._matrix is None or not question.strip():
            return []
        query_vector = self._vectorizer.transform([question])
        scores = cosine_similarity(query_vector, self._matrix).ravel()
        best_first = scores.argsort()[::-1][:top_k]
        return [
            SearchResult(self.chunks[i], float(scores[i]))
            for i in best_first
            if scores[i] >= min_score
        ]
