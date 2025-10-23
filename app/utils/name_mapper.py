"""
Pokemon Name Mapper

Utility for translating Korean Pokemon names to English.
"""
import json
import unicodedata
from pathlib import Path
from typing import Dict, Optional


class PokemonNameMapper:
    """Maps Korean Pokemon names to English names"""

    def __init__(self, json_path: Optional[str] = None):
        """
        Initialize name mapper

        Args:
            json_path: Path to pokemon.names.json file
        """
        if json_path is None:
            # Default path relative to this file
            base_dir = Path(__file__).resolve().parent.parent
            json_path = base_dir / "ai_models" / "pokemon" / "pokemon.names.json"

        self.name_mapping: Dict[str, str] = {}
        self._load_mappings(json_path)

    def _normalize_key(self, text: str) -> str:
        """
        Normalize Korean text to NFC form for consistent key matching

        Args:
            text: Input text (may be in NFD or NFC form)

        Returns:
            Normalized text in NFC form (composed characters)
        """
        return unicodedata.normalize('NFC', text)

    def _load_mappings(self, json_path: Path):
        """Load name mappings from JSON file"""
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                raw_mapping = json.load(f)

            # Normalize all keys to NFC form
            self.name_mapping = {
                self._normalize_key(key): value
                for key, value in raw_mapping.items()
            }
            print(f"✓ Loaded {len(self.name_mapping)} Pokemon name mappings")
        except FileNotFoundError:
            print(f"⚠ Warning: Pokemon names file not found at {json_path}")
            self.name_mapping = {}
        except json.JSONDecodeError as e:
            print(f"⚠ Warning: Failed to parse Pokemon names JSON: {e}")
            self.name_mapping = {}

    def to_english(self, korean_name: str) -> str:
        """
        Convert Korean name to English

        Args:
            korean_name: Korean Pokemon name (will be normalized)

        Returns:
            English name if found, otherwise original Korean name
        """
        # Normalize input key to match stored keys
        normalized_name = self._normalize_key(korean_name)
        return self.name_mapping.get(normalized_name, korean_name)

    def translate_verdict(self, verdict: str) -> str:
        """
        Translate verdict (special handling for 'unknown')

        Args:
            verdict: Verdict string (class name or 'unknown')

        Returns:
            Translated verdict
        """
        if verdict.lower() == "unknown":
            return "unknown"
        return self.to_english(verdict)

    def has_mapping(self, korean_name: str) -> bool:
        """
        Check if mapping exists for given name

        Args:
            korean_name: Korean name to check (will be normalized)

        Returns:
            True if mapping exists
        """
        normalized_name = self._normalize_key(korean_name)
        return normalized_name in self.name_mapping
