"""
Similarity API Endpoints
"""
from fastapi import APIRouter, Depends, UploadFile, File, Query, HTTPException
from app.schemas.similarity import ImageUrlRequest, SimilarityResponse
from app.services.embedding_service import EmbeddingService
from app.services.similarity_service import SimilarityService
from app.core.dependencies import get_embedding_extractor, get_similarity_calculator

router = APIRouter(prefix="/similarity", tags=["similarity"])


@router.post("/find-by-upload", response_model=SimilarityResponse)
async def find_similar_by_upload(
    file: UploadFile = File(..., description="Image file to analyze"),
    top_k: int = Query(3, ge=1, le=10, description="Number of top similar results to return"),
    extractor=Depends(get_embedding_extractor),
    calculator=Depends(get_similarity_calculator)
):
    """
    Find similar Pokémon from uploaded image file

    - **file**: Image file (multipart/form-data)
    - **top_k**: Number of top results (default: 3, range: 1-10)
    """
    try:
        # Initialize services
        embedding_service = EmbeddingService(extractor)
        similarity_service = SimilarityService(calculator)

        # Extract embedding from uploaded file
        embedding = await embedding_service.extract_from_upload(file)

        # Find similar
        result = similarity_service.find_similar(embedding, top_k=top_k)

        return SimilarityResponse(**result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@router.post("/find-by-url", response_model=SimilarityResponse)
async def find_similar_by_url(
    request: ImageUrlRequest,
    top_k: int = Query(3, ge=1, le=10, description="Number of top similar results to return"),
    extractor=Depends(get_embedding_extractor),
    calculator=Depends(get_similarity_calculator)
):
    """
    Find similar Pokémon from image URL

    - **url**: Image URL (HTTP/HTTPS)
    - **top_k**: Number of top results (default: 3, range: 1-10)
    """
    try:
        # Initialize services
        embedding_service = EmbeddingService(extractor)
        similarity_service = SimilarityService(calculator)

        # Extract embedding from URL
        embedding = await embedding_service.extract_from_url(str(request.url))

        # Find similar
        result = similarity_service.find_similar(embedding, top_k=top_k)

        print('URL:', request.url)
        print('Result:', result)

        return SimilarityResponse(**result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
