"""
Embedding Extraction Service
"""
import numpy as np
from PIL import Image
from typing import Union
from fastapi import UploadFile
import io
import requests
from app.models.embedding_extractor import ImageEmbeddingExtractor


class EmbeddingService:
    """Service for extracting image embeddings"""

    def __init__(self, extractor: ImageEmbeddingExtractor):
        self.extractor = extractor

    async def extract_from_upload(self, file: UploadFile) -> np.ndarray:
        """
        Extract embedding from uploaded file

        Args:
            file: Uploaded image file

        Returns:
            Embedding vector as numpy array
        """
        # Read uploaded file
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')

        # Extract embedding using the extract_embedding method which accepts PIL Image
        embedding = self.extractor.extract_embedding(image)

        return embedding

    async def extract_from_url(self, url: str) -> np.ndarray:
        """
        Extract embedding from image URL

        Args:
            url: Image URL

        Returns:
            Embedding vector as numpy array
        """
        # Download image
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        # Open image
        image = Image.open(io.BytesIO(response.content)).convert('RGB')

        # Extract embedding using the extract_embedding method which accepts PIL Image
        embedding = self.extractor.extract_embedding(image)

        return embedding
