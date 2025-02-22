"""KV Cache implementation with SSD storage."""

import os
from pathlib import Path
import torch

class KVCache:
    def __init__(self, cache_dir: str = "kv_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def save_kv_cache_to_ssd(self, cache_data: dict, cache_name: str) -> None:
        """Save KV cache data to SSD."""
        os.makedirs(os.path.dirname(self.cache_dir), exist_ok=True) # Ensure dir exists
        cache_path = self.cache_dir / f"{cache_name}.pt"
        torch.save(cache_data, cache_path)
        print(f"KV Cache saved to: {cache_path}")
    
    def load_kv_cache_from_ssd(self, cache_name: str) -> dict:
        """Load KV cache data from SSD."""
        cache_path = self.cache_dir / f"{cache_name}.pt"
        print(f"Loading KV Cache from: {cache_path}")
        if os.path.exists(cache_path):
            cache_data = torch.load(cache_path, map_location=torch.device('cpu')) # Load to CPU initially
            print("KV Cache loaded successfully.")
            return cache_data
        else:
            print("KV Cache file not found. Please preprocess and create the cache first.")
            return None
