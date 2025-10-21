"""
Application Configuration
"""
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""

    # Application
    app_name: str = "Image Similarity API"
    app_version: str = "1.0.0"

    # Paths
    base_dir: Path = Path(__file__).resolve().parent.parent
    ai_models_dir: Path = base_dir / "ai_models" / "pokemon"
    prototypes_path: Path = ai_models_dir / "prototypes.npz"

    # Model settings
    model_name: str = "openai/clip-vit-base-patch32"
    device: str = None  # None for auto-detection (cuda/cpu)

    # Similarity thresholds
    tau1: float = 0.30  # Minimum top similarity score threshold
    tau2: float = 0.04  # Minimum margin threshold

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
