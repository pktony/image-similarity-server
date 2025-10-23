"""
Application Configuration
"""
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings"""

    # Application
    app_name: str = "Image Similarity API"
    app_version: str = "1.0.0"

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000

    # Paths
    base_dir: Path = Path(__file__).resolve().parent.parent
    ai_models_dir: Path = base_dir / "ai_models" / "pokemon"
    prototypes_path: Path = ai_models_dir / "prototypes.npz"

    # Model settings
    model_name: str = "openai/clip-vit-base-patch32"
    device: Optional[str] = None  # None for auto-detection (cuda/cpu)

    # Similarity thresholds
    tau1: float = 0.30  # Minimum top similarity score threshold
    tau2: float = 0.04  # Minimum margin threshold

    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).resolve().parent.parent.parent / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )


settings = Settings()

# Debug: Print loaded configuration
print(f"[CONFIG] .env file path: {Path(__file__).resolve().parent.parent.parent / '.env'}")
print(f"[CONFIG] .env exists: {(Path(__file__).resolve().parent.parent.parent / '.env').exists()}")
print(f"[CONFIG] Loaded PORT: {settings.port}")
print(f"[CONFIG] Loaded HOST: {settings.host}")
