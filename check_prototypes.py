"""
Check prototypes.npz keys and compare with pokemon.names.json
"""
import numpy as np
import json
from pathlib import Path
import sys
import io

# Set stdout to UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Load prototypes
prototypes_path = Path("app/ai_models/pokemon/prototypes.npz")
names_path = Path("app/ai_models/pokemon/pokemon.names.json")

print("=== Checking Prototypes Keys ===")
data = np.load(prototypes_path)
prototype_keys = set(data.files)
print(f"Total prototype keys: {len(prototype_keys)}")
print(f"Sample keys: {list(prototype_keys)[:10]}")
print()

print("=== Checking pokemon.names.json Keys ===")
with open(names_path, "r", encoding="utf-8") as f:
    name_mapping = json.load(f)
mapping_keys = set(name_mapping.keys())
print(f"Total mapping keys: {len(mapping_keys)}")
print(f"Sample keys: {list(mapping_keys)[:10]}")
print()

print("=== Comparing Keys ===")
in_prototypes_not_in_mapping = prototype_keys - mapping_keys
in_mapping_not_in_prototypes = mapping_keys - prototype_keys

if in_prototypes_not_in_mapping:
    print(f"⚠ Keys in prototypes but NOT in mapping ({len(in_prototypes_not_in_mapping)}):")
    for key in sorted(in_prototypes_not_in_mapping):
        print(f"  - {repr(key)}")
else:
    print("✓ All prototype keys are in mapping")

print()

if in_mapping_not_in_prototypes:
    print(f"⚠ Keys in mapping but NOT in prototypes ({len(in_mapping_not_in_prototypes)}):")
    for key in sorted(in_mapping_not_in_prototypes):
        print(f"  - {repr(key)}")
else:
    print("✓ All mapping keys are in prototypes")

print()
print("=== Key Character Analysis ===")
if prototype_keys:
    sample_key = list(prototype_keys)[0]
    print(f"Sample prototype key: {repr(sample_key)}")
    print(f"  Length: {len(sample_key)}")
    print(f"  Bytes: {sample_key.encode('utf-8')}")
    print(f"  Characters: {[ord(c) for c in sample_key]}")

if mapping_keys:
    sample_key = list(mapping_keys)[0]
    print(f"Sample mapping key: {repr(sample_key)}")
    print(f"  Length: {len(sample_key)}")
    print(f"  Bytes: {sample_key.encode('utf-8')}")
    print(f"  Characters: {[ord(c) for c in sample_key]}")

print()
print("=== Testing Unicode Normalization ===")
import unicodedata

if prototype_keys and mapping_keys:
    proto_key = list(prototype_keys)[0]
    map_key = list(mapping_keys)[0]

    proto_nfc = unicodedata.normalize('NFC', proto_key)
    map_nfc = unicodedata.normalize('NFC', map_key)

    print(f"Prototype key normalized (NFC): {repr(proto_nfc)} (len={len(proto_nfc)})")
    print(f"Mapping key normalized (NFC): {repr(map_nfc)} (len={len(map_nfc)})")

    # Test matching after normalization
    normalized_proto_keys = {unicodedata.normalize('NFC', k) for k in prototype_keys}
    normalized_map_keys = {unicodedata.normalize('NFC', k) for k in mapping_keys}

    matches = normalized_proto_keys & normalized_map_keys
    print(f"✓ After normalization: {len(matches)} matching keys out of {len(prototype_keys)} total")

    if len(matches) == len(prototype_keys):
        print("✓✓ Perfect match! All keys match after normalization.")
