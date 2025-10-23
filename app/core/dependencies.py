"""
Dependency Injection
"""
from typing import Optional
from app.models.embedding_extractor import ImageEmbeddingExtractor
from app.models.similarity_calculator import SimilarityCalculator
from app.utils.name_mapper import PokemonNameMapper
from app.core.config import settings


class ModelManager:
    """Singleton manager for AI models"""

    _instance: Optional["ModelManager"] = None
    _embedding_extractor: Optional[ImageEmbeddingExtractor] = None
    _similarity_calculator: Optional[SimilarityCalculator] = None
    _name_mapper: Optional[PokemonNameMapper] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def initialize(self):
        """Initialize models on app startup"""
        if self._embedding_extractor is None:
            print(f"Loading embedding extractor: {settings.model_name}")
            self._embedding_extractor = ImageEmbeddingExtractor(
                model_name=settings.model_name,
                device=settings.device
            )

        if self._name_mapper is None:
            print(f"Loading Pokemon name mapper from: {settings.pokemon_names_path}")
            self._name_mapper = PokemonNameMapper(str(settings.pokemon_names_path))

        if self._similarity_calculator is None:
            print(f"Loading similarity calculator from: {settings.prototypes_path}")
            self._similarity_calculator = SimilarityCalculator(
                tau1=settings.tau1,
                tau2=settings.tau2,
                name_mapper=self._name_mapper
            )
            self._similarity_calculator.load_prototypes_from_npz(
                str(settings.prototypes_path)
            )

        print("✓ All models loaded successfully")

    @property
    def embedding_extractor(self) -> ImageEmbeddingExtractor:
        """Get embedding extractor instance"""
        if self._embedding_extractor is None:
            raise RuntimeError("ModelManager not initialized. Call initialize() first.")
        return self._embedding_extractor

    @property
    def similarity_calculator(self) -> SimilarityCalculator:
        """Get similarity calculator instance"""
        if self._similarity_calculator is None:
            raise RuntimeError("ModelManager not initialized. Call initialize() first.")
        return self._similarity_calculator


# Global model manager instance
model_manager = ModelManager()


def get_embedding_extractor() -> ImageEmbeddingExtractor:
    """Dependency for embedding extractor"""
    return model_manager.embedding_extractor


def get_similarity_calculator() -> SimilarityCalculator:
    """Dependency for similarity calculator"""
    return model_manager.similarity_calculator
