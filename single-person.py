import os
import numpy as np
import sqlite3
import base64


def decode_embedding(embedding: str, secret_key: str = "I2jhEJRStK") -> np.ndarray:
    return np.frombuffer(base64.b64decode(embedding.replace(secret_key, '')), dtype=np.float32)


DB_FILE = "resources/centroid-embedding.db"
TABLE_NAME = "centroids"


def normalize(embedding):
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return embedding
    return embedding / norm


def cosine_sim(a, b):
    return np.dot(normalize(a), normalize(b))


def load_centroid_from_db(person_code, db_file=DB_FILE):
    """
    Load centroid embedding for a specific person_code from the SQLite database.
    Returns: dict {'centroid_embedding': np.array, ...} or None if not found.
    """
    if not os.path.exists(db_file):
        print(f"❌ Centroid DB file not found: {db_file}")
        return None
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    try:
        c.execute(
            f"SELECT centroid_embedding, num_features, date_processed FROM {TABLE_NAME} WHERE person_code = ?",
            (person_code,)
        )
        row = c.fetchone()
        if row is None:
            print(f"❌ person_code '{person_code}' not found in centroid DB")
            return None
        emb_bytes, num_features, date_processed = row
        emb = np.frombuffer(emb_bytes, dtype=np.float32)
        return {
            'centroid_embedding': emb,
            'num_features': num_features,
            'date_processed': date_processed
        }
    except Exception as e:
        print(f"❌ Error loading centroid from DB: {e}")
        return None
    finally:
        conn.close()


def compare_feature_with_centroid(feature_string, secret_key, person_code, db_file=DB_FILE, threshold=0.5):
    """
    Given a feature string, compare with centroid embedding of person_code from SQLite DB.
    Returns: dict with scores and info.
    """
    # Load centroid embedding for the person_code
    centroid_data = load_centroid_from_db(person_code, db_file=db_file)
    if centroid_data is None:
        return None

    # Decode embedding
    try:
        embedding = decode_embedding(feature_string, secret_key)
        test_embedding = normalize(embedding)
    except Exception as e:
        print(f"❌ Error decoding feature: {e}")
        return None

    # Get centroid embedding
    centroid_embedding = centroid_data['centroid_embedding']
    centroid_embedding = normalize(centroid_embedding)

    # Calculate similarity
    score = cosine_sim(test_embedding, centroid_embedding)

    result = {
        'person_code': person_code,
        'score': float(score),
        'threshold': threshold,
        'is_match': score >= threshold,
        'num_features': centroid_data.get('num_features'),
        'date_processed': centroid_data.get('date_processed')
    }
    print(f"Comparison result for person_code={person_code}:")
    print(f"  similarity score: {score:.4f}")
    print(f"  threshold: {result['threshold']}")
    print(f"  is_match: {result['is_match']}")
    return result


if __name__ == "__main__":
    FEATURE = "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"
    PERSON_CODE = "3000021649"
    SECRET_KEY = "I2jhEJRStK"

    result = compare_feature_with_centroid(
        feature_string=FEATURE,
        db_file=DB_FILE,
        secret_key=SECRET_KEY,
        person_code=PERSON_CODE
    )
