"""
Oracle Object Storage Service
Handles PAR (Pre-authenticated Request) URL generation
"""
import os
from datetime import datetime, timedelta
from typing import Dict
import oci
from app.core.config import settings


class OracleStorageService:
    """Service for Oracle Object Storage operations"""

    def __init__(self):
        """Initialize OCI client using config file"""
        # Load OCI config from ~/.oci/config
        config_file = os.path.expanduser("~/.oci/config")
        self.config = oci.config.from_file(
            file_location=config_file,
            profile_name=settings.oci_config_profile
        )

        # Initialize Object Storage client
        self.client = oci.object_storage.ObjectStorageClient(self.config)
        print('oracle_storage_service: initialized OCI client')

        # Get namespace (if not provided in settings)
        if settings.oci_namespace:
            self.namespace = settings.oci_namespace
        else:
            # Auto-detect namespace
            self.namespace = self.client.get_namespace().data

        self.bucket_name = settings.oci_bucket_name
        self.region = settings.oci_region

        print('oracle_storage_service: bucket_name ' + self.bucket_name)
        print('oracle_storage_service: region ' + self.region)

    def generate_upload_url(self, filename: str) -> Dict[str, str]:
        """
        Generate Pre-authenticated Request URL for file upload

        Args:
            filename: Name of the file to upload

        Returns:
            Dictionary containing:
            - upload_url: PUT request URL for uploading the file
            - download_url: GET request URL for accessing the uploaded file
            - expires_at: ISO format expiration timestamp
        """
        # Sanitize filename (prevent path traversal)
        safe_filename = os.path.basename(filename)

        # Generate unique object name (timestamp prefix)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        object_name = f"{timestamp}_{safe_filename}"

        # Set PAR expiration (1 hour from now)
        expires_at = datetime.utcnow() + timedelta(hours=1)

        # Create PAR for uploading (PUT access)
        par_details = oci.object_storage.models.CreatePreauthenticatedRequestDetails(
            name=f"upload_{object_name}",
            access_type="ObjectWrite",  # PUT only
            time_expires=expires_at,
            object_name=object_name,
            bucket_listing_action=None
        )

        # Generate PAR
        par_response = self.client.create_preauthenticated_request(
            namespace_name=self.namespace,
            bucket_name=self.bucket_name,
            create_preauthenticated_request_details=par_details
        )

        # Build full PAR URL
        upload_url = f"https://objectstorage.{self.region}.oraclecloud.com{par_response.data.access_uri}"

        # Build download URL (public access URL without PAR)
        download_url = (
            f"https://objectstorage.{self.region}.oraclecloud.com"
            f"/n/{self.namespace}/b/{self.bucket_name}/o/{object_name}"
        )

        return {
            "upload_url": upload_url,
            "download_url": download_url,
            "object_name": object_name,
            "expires_at": expires_at.isoformat() + "Z"
        }

    def delete_object(self, object_name: str) -> None:
        """
        Delete an object from the bucket (optional cleanup)

        Args:
            object_name: Name of the object to delete
        """
        self.client.delete_object(
            namespace_name=self.namespace,
            bucket_name=self.bucket_name,
            object_name=object_name
        )
