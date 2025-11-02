"""
Upload API Endpoints v2
Pre-authenticated Request URL generation for Oracle Object Storage
"""
import oci
from fastapi import APIRouter, HTTPException
from app.schemas.upload import UploadUrlRequest, UploadUrlResponse
from app.services.oracle_storage_service import OracleStorageService

router = APIRouter(prefix="/upload", tags=["upload-v2"])


@router.post("/generate-url", response_model=UploadUrlResponse)
async def generate_upload_url(request: UploadUrlRequest):
    """
    Generate Pre-authenticated Request URL for image upload

    Workflow:
    1. Receive filename from frontend
    2. Generate PAR URL for uploading to Oracle Object Storage
    3. Return upload URL (PUT), download URL (GET), and expiration time

    Frontend should:
    - Use `upload_url` to upload the file via PUT request
    - Use `download_url` to send to /api/v2/similarity/analyze

    - **filename**: Name of the file to upload (e.g., 'pikachu.jpg')

    Returns:
    - upload_url: URL for uploading file (PUT method)
    - download_url: URL for accessing uploaded file
    - object_name: Generated object name in storage
    - expires_at: PAR expiration timestamp (1 hour validity)
    """
    try:
        # Initialize Oracle Storage service
        storage_service = OracleStorageService()

        # Generate PAR URL
        result = storage_service.generate_upload_url(request.filename)

        return UploadUrlResponse(**result)

    except oci.exceptions.ServiceError as e:
        # OCI-specific errors
        raise HTTPException(
            status_code=500,
            detail=f"Oracle Cloud error: {e.message}"
        )
    except Exception as e:
        # Generic errors
        raise HTTPException(
            status_code=500,
            detail=f"Error generating upload URL: {str(e)}"
        )
