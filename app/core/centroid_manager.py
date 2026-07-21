import os
import sqlite3
import numpy as np
from typing import Optional, List, Dict, Any


class CentroidManager:
    """Manager for handling centroid embeddings using SQLite."""

    def __init__(self):
        self.db_file = "resources/centroid-embedding.db"
        self.table_name = "centroids"
        self.centroid_embeddings = {}
        self._create_db_and_table()
        self._load_centroid_embeddings()

    def _create_db_and_table(self):
        """Create database and table if they don't exist."""
        conn = sqlite3.connect(self.db_file)
        c = conn.cursor()
        c.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                person_code TEXT PRIMARY KEY,
                centroid_embedding BLOB NOT NULL,
                num_features INTEGER,
                date_processed TEXT
            )
        """)
        conn.commit()
        conn.close()

    def _load_centroid_embeddings(self):
        """Load centroid embeddings from SQLite database."""
        try:
            if not os.path.exists(self.db_file):
                print(f"Centroid database file not found at {self.db_file}")
                self.centroid_embeddings = {}
                return

            conn = sqlite3.connect(self.db_file)
            c = conn.cursor()
            c.execute(
                f"SELECT person_code, centroid_embedding, num_features, date_processed FROM {self.table_name}")
            rows = c.fetchall()

            for row in rows:
                person_code, emb_bytes, num_features, date_processed = row
                # Assume 512-dim float32
                emb = np.frombuffer(emb_bytes, dtype=np.float32)
                self.centroid_embeddings[person_code] = {
                    'centroid_embedding': emb,
                    'num_features': num_features,
                    'date_processed': date_processed
                }

            conn.close()

            print(
                f"Loaded {len(self.centroid_embeddings)} centroid embeddings from {self.db_file}")

            # Print some statistics
            if self.centroid_embeddings:
                sample_person = list(self.centroid_embeddings.keys())[0]
                sample_data = self.centroid_embeddings[sample_person]
                print(f"Sample centroid data for {sample_person}:")
                print(
                    f"  - Number of features used: {sample_data.get('num_features', 'N/A')}")
                print(
                    f"  - Date processed: {sample_data.get('date_processed', 'N/A')}")
                print(
                    f"  - Centroid embedding shape: {sample_data.get('centroid_embedding', np.array([])).shape}")
        except Exception as e:
            print(f"Error loading centroid embeddings: {e}")
            self.centroid_embeddings = {}

    def has_centroid_embedding(self, person_code: str) -> bool:
        """Check if a person has centroid embedding available."""
        return person_code in self.centroid_embeddings

    def get_centroid_embedding(self, person_code: str) -> Optional[np.ndarray]:
        """Get centroid embedding for a person code."""
        if person_code in self.centroid_embeddings:
            return self.centroid_embeddings[person_code]['centroid_embedding']
        return None

    def get_centroid_info(self, person_code: str) -> Optional[Dict[str, Any]]:
        """Get centroid information for a person code."""
        return self.centroid_embeddings.get(person_code)

    def get_all_centroid_codes(self) -> List[str]:
        """Get all person codes that have centroid embeddings."""
        return list(self.centroid_embeddings.keys())

    def get_centroid_count(self) -> int:
        """Get total number of centroid embeddings loaded."""
        return len(self.centroid_embeddings)

    def get_centroid_statistics(self) -> Dict[str, Any]:
        """Get statistics about loaded centroid embeddings."""
        if not self.centroid_embeddings:
            return {
                "total_centroids": 0,
                "sample_person_code": None,
                "sample_info": None,
                "all_person_codes": [],
                "has_more": False
            }

        all_codes = self.get_all_centroid_codes()
        sample_code = all_codes[0] if all_codes else None
        sample_info = self.get_centroid_info(
            sample_code) if sample_code else None

        return {
            "total_centroids": self.get_centroid_count(),
            "sample_person_code": sample_code,
            "sample_info": sample_info,
            "all_person_codes": all_codes[:10] if len(all_codes) > 10 else all_codes,
            "has_more": len(all_codes) > 10
        }

    def reload_centroid_embeddings(self):
        """Reload centroid embeddings from database."""
        print("Reloading centroid embeddings from database...")
        self._load_centroid_embeddings()

    def is_centroid_available(self) -> bool:
        """Check if centroid embeddings are available."""
        return len(self.centroid_embeddings) > 0


# Create a singleton instance
centroid_manager = CentroidManager()
