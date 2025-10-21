"""
Similarity Calculation Service
"""
import numpy as np
from typing import Dict
from app.models.similarity_calculator import SimilarityCalculator


class SimilarityService:
    """Service for calculating similarity between embeddings"""

    def __init__(self, calculator: SimilarityCalculator):
        self.calculator = calculator

    def find_similar(self, query_embedding: np.ndarray, top_k: int = 3) -> Dict:
        """
        Find similar classes for given embedding

        Args:
            query_embedding: Query embedding vector
            top_k: Number of top results to return

        Returns:
            Dictionary with match results:
            {
                "top_k": [(name, score), ...],
                "verdict": str,
                "s1": float,
                "margin": float,
                "is_unknown": bool
            }
        """
        result = self.calculator.match(query_embedding, k=top_k)
        return result
