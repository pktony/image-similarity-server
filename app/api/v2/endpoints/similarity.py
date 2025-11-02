"""
Similarity API Endpoints v2
Oracle Pre-authenticated Request optimized version
"""
from fastapi import APIRouter, Depends, Query, HTTPException
from app.schemas.similarity import ImageUrlRequest, SimilarityResponse
from app.services.embedding_service import EmbeddingService
from app.services.similarity_service import SimilarityService
from app.core.dependencies import get_embedding_extractor, get_similarity_calculator

router = APIRouter(prefix="/similarity", tags=["similarity-v2"])


@router.post("/analyze", response_model=SimilarityResponse)
async def analyze_image_from_url(
    request: ImageUrlRequest,
    top_k: int = Query(3, ge=1, le=10, description="Number of top similar results to return"),
    extractor=Depends(get_embedding_extractor),
    calculator=Depends(get_similarity_calculator)
):
    """
    Analyze image from URL (optimized for Oracle Pre-authenticated Request)

    Workflow:
    1. Frontend uploads image to Oracle Object Storage using Pre-authenticated Request
    2. Frontend sends the uploaded image URL to this endpoint
    3. Server downloads image from URL, extracts embedding, and calculates similarity

    - **url**: Image URL (Oracle Object Storage Pre-authenticated Request URL)
    - **top_k**: Number of top results (default: 3, range: 1-10)

    Returns:
    - Top K similar Pokémon with similarity scores
    - Verdict (most similar class or 'unknown')
    - Similarity metrics (s1, margin, is_unknown)
    """
    try:
        # Initialize services
        embedding_service = EmbeddingService(extractor)
        similarity_service = SimilarityService(calculator)

        # Extract embedding from URL
        embedding = await embedding_service.extract_from_url(str(request.url))

        # Find similar
        result = similarity_service.find_similar(embedding, top_k=top_k)

        return SimilarityResponse(**result)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image from URL: {str(e)}"
        )
