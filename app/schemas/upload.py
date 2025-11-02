"""
Pydantic schemas for upload API
"""
from pydantic import BaseModel, Field


class UploadUrlRequest(BaseModel):
    """Request schema for generating upload URL"""
    filename: str = Field(
        ...,
        description="Name of the file to upload (e.g., 'pikachu.jpg')",
        min_length=1,
        max_length=255
    )

    class Config:
        json_schema_extra = {
            "example": {
                "filename": "pikachu.jpg"
            }
        }


class UploadUrlResponse(BaseModel):
    """Response schema for upload URL generation"""
    upload_url: str = Field(
        ...,
        description="Pre-authenticated Request URL for uploading (PUT method)"
    )
    download_url: str = Field(
        ...,
        description="URL for downloading/accessing the uploaded file"
    )
    object_name: str = Field(
        ...,
        description="Generated object name in storage (timestamp_filename)"
    )
    expires_at: str = Field(
        ...,
        description="PAR expiration time in ISO 8601 format"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "upload_url": "https://objectstorage.ap-seoul-1.oraclecloud.com/p/abc123.../n/namespace/b/bucket/o/20251102_120000_pikachu.jpg",
                "download_url": "https://objectstorage.ap-seoul-1.oraclecloud.com/n/namespace/b/bucket/o/20251102_120000_pikachu.jpg",
                "object_name": "20251102_120000_pikachu.jpg",
                "expires_at": "2025-11-02T13:00:00Z"
            }
        }
