from typing import List, Optional, Tuple
import asyncio
import numpy as np
import torch
import json
import os
from app.core.config import app_config
from app.utils.helpers import decode_embedding
from app.utils.helpers import cosine_similarity
from app.schemas.analyze.search_face import FaceResult
from app.core.local_db import local_db_manager
from app.core.centroid_manager import centroid_manager


def _get_person_type(person: dict):
    if 'personType' in person:
        return person.get('personType')
    return person.get('person_type')


class DatabaseManager:
    def __init__(self):
        self._lock = asyncio.Lock()
        self.persons = []
        self.face_features_matrix = None
        self.person_indices = []
        self.hard_to_recognize_codes = set()
        self.use_gpu = torch.cuda.is_available()
        if self.use_gpu:
            print("GPU available - using CUDA for face search")
        else:
            print("GPU not available - using CPU with numpy vectorization")

        # Load hard to recognize persons
        self._load_hard_to_recognize_persons()

    def _load_hard_to_recognize_persons(self):
        """Load list of hard to recognize persons from JSON file."""
        try:
            config_path = os.path.join(
                "resources", "config", "hard_to_recognize.json")
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    hard_persons = data.get("hard_to_recognize_persons", [])
                    self.hard_to_recognize_codes = {
                        person["personCode"] for person in hard_persons}
                    print(
                        f"Loaded {len(self.hard_to_recognize_codes)} hard to recognize persons: {list(self.hard_to_recognize_codes)}")
            else:
                print(
                    f"Hard to recognize config file not found at {config_path}")
        except Exception as e:
            print(f"Error loading hard to recognize persons: {e}")

    async def load_persons(self):
        """
        Load person data from local SQLite database and prepare for vectorized search.

        Returns:
            List of person records as dictionaries
        """
        async with self._lock:
            try:
                self.persons = await local_db_manager.get_all_persons()

                if not self.persons:
                    print("No persons found in database")
                    return []

                # Decode embeddings and prepare for vectorized search
                face_features = []
                self.person_indices = []
                print("Total person processing: ", len(self.persons))
                for i, person in enumerate(self.persons):
                    try:
                        face_feature = decode_embedding(
                            person['faceFeature'], app_config.SECRET_KEY)
                        face_features.append(face_feature)
                        self.person_indices.append(i)
                        person['FaceFeature'] = face_feature
                    except Exception as e:
                        print(
                            f"Error decoding embedding for person {person['personId']}: {e}")
                        continue

                if face_features:
                    # Filter embeddings to only include 512-dimensional ones
                    target_dim = 512
                    filtered_features = []
                    filtered_indices = []

                    for i, feature in enumerate(face_features):
                        if feature.shape[0] == target_dim:
                            filtered_features.append(feature)
                            filtered_indices.append(self.person_indices[i])
                        else:
                            person_idx = self.person_indices[i]
                            print(
                                f"Removing person {self.persons[person_idx]['personId']}: embedding dimension {feature.shape[0]} != {target_dim}")

                    if filtered_features:
                        face_features = filtered_features
                        self.person_indices = filtered_indices
                        print(
                            f"Kept {len(filtered_features)} embeddings with dimension {target_dim}")
                    else:
                        print(
                            f"Error: No embeddings with dimension {target_dim} found!")
                        return []

                    # Convert to numpy array for vectorized operations
                    self.face_features_matrix = np.array(
                        face_features, dtype=np.float32)

                    # Move to GPU if available
                    if self.use_gpu:
                        self.face_features_matrix = torch.from_numpy(
                            self.face_features_matrix).cuda()

                    print(
                        f"Loaded {len(self.persons)} persons from local database")
                    print(
                        f"Prepared {len(face_features)} face features for vectorized search")
                else:
                    print("No valid face features found")

                return self.persons
            except Exception as e:
                print(f"Error loading persons from local database: {e}")
                return []

    def _compute_similarities_vectorized(self, face_embedding: np.ndarray, company_ids: Optional[List[int]] = None) -> Tuple[np.ndarray, List[int]]:
        """
        Compute similarities using vectorized operations.

        Args:
            face_embedding: Face feature vector to search for
            company_ids: Optional list of company IDs to filter by

        Returns:
            Tuple of (similarities, valid_indices)
        """
        if self.face_features_matrix is None:
            return np.array([]), []

        # Build list of row indices into face_features_matrix.
        # self.person_indices maps row_index -> index in self.persons.
        valid_row_indices: List[int] = []
        if company_ids:
            company_ids_set = set(company_ids)
            for row_idx, person_idx in enumerate(self.person_indices):
                person = self.persons[person_idx]
                company_id = (
                    person.get('company_id')
                    if person.get('company_id') is not None
                    else person.get('compId')
                )
                # Nếu company_id bị thiếu (None) thì vẫn cho phép search, không filter out.
                if company_id is None or company_id in company_ids_set:
                    valid_row_indices.append(row_idx)
        else:
            valid_row_indices = list(range(len(self.person_indices)))

        if not valid_row_indices:
            return np.array([]), []

        # Get filtered face features
        if self.use_gpu:
            # GPU computation with torch
            face_embedding_tensor = torch.from_numpy(
                face_embedding.astype(np.float32)).cuda()
            filtered_features = self.face_features_matrix[valid_row_indices]

            # Compute cosine similarity using torch
            similarities = torch.nn.functional.cosine_similarity(
                face_embedding_tensor.unsqueeze(0),
                filtered_features,
                dim=1
            )
            similarities = similarities.cpu().numpy()
        else:
            # CPU computation with numpy vectorization
            filtered_features = self.face_features_matrix[valid_row_indices]

            # Normalize vectors for cosine similarity
            face_embedding_norm = face_embedding / \
                np.linalg.norm(face_embedding)
            features_norm = filtered_features / \
                np.linalg.norm(filtered_features, axis=1, keepdims=True)

            # Compute dot product (cosine similarity)
            similarities = np.dot(features_norm, face_embedding_norm)

        return similarities, valid_row_indices

    async def search_person(self, face_embedding: np.ndarray, threshold: float = 0.5, company_ids: Optional[List[int]] = None) -> Tuple[List[FaceResult], Optional[FaceResult]]:
        """
        Search for matching faces in the database using vectorized cosine similarity.

        Args:
            face_embedding: Face feature vector to search for
            threshold: Similarity threshold (default: 0.5)
            company_ids: Optional list of company IDs to filter by

        Returns:
            Tuple of (recognized_persons, nearest_person) - only returns the best match
        """
        if not self.persons:
            print("Waiting for persons to load from local database")
            await self.load_persons()

        if not self.persons or self.face_features_matrix is None:
            print("No persons loaded or face features not prepared")
            return [], None

        # Validate embedding dimension
        if not self.validate_embedding(face_embedding):
            print("Invalid embedding dimension. Expected 512-dimensional embedding.")
            return [], None
        threshold /= 2
        similarity_threshold = app_config.SIMILARITY_THRESHOLD / 2
        centroid_threshold = app_config.CENTROID_THRESHOLD / 2
        try:
            # Compute similarities using vectorized operations
            similarities, valid_indices = self._compute_similarities_vectorized(
                face_embedding, company_ids)

            if len(similarities) == 0:
                print("No valid persons found matching criteria")
                return [], None

            # Find the best similarity using numpy argmax
            best_similarity_index = np.argmax(similarities)
            best_similarity = similarities[best_similarity_index]
            best_person_idx = self.person_indices[valid_indices[best_similarity_index]]
            best_person = self.persons[best_person_idx]

            nearest_person = FaceResult(
                personId=best_person['personId'],
                personCode=best_person['personCode'],
                fullname=best_person['fullname'],
                department_id=str(best_person.get('deptName') or best_person.get('department_id') or ''),
                person_type=_get_person_type(best_person),
                scoreMatching=float(best_similarity)
            )

            # Check if the best match is a hard to recognize person
            if nearest_person and nearest_person.personCode in self.hard_to_recognize_codes:
                print(
                    f"Hard to recognize person detected: {nearest_person.fullname} ({nearest_person.personCode})")
                # Use fixed threshold of 0.7 for hard to recognize persons
                hard_threshold = 0.35
                if nearest_person.scoreMatching >= hard_threshold:
                    nearest_person.scoreMatching = min(
                        nearest_person.scoreMatching * 2, 1)
                    print(
                        f"Found hard to recognize person: {nearest_person.fullname} with score {nearest_person.scoreMatching:.3f}")
                    return [nearest_person], nearest_person
                else:
                    print(
                        f"Hard to recognize person {nearest_person.fullname} below threshold {hard_threshold}")
                    return [], None

            # Check if the best match is a centroid person
            if nearest_person and nearest_person.personCode in centroid_manager.get_all_centroid_codes():
                print("###Centroid person detected: ",
                      nearest_person.fullname)
                centroid_embedding = centroid_manager.get_centroid_embedding(
                    nearest_person.personCode)
                if centroid_embedding is not None:
                    centroid_similarity = cosine_similarity(
                        face_embedding, centroid_embedding)
                    if centroid_similarity >= centroid_threshold and nearest_person.scoreMatching >= threshold:
                        print("###Centroid person found")
                        print(
                            f"Got person {nearest_person.personCode} with centroid {centroid_similarity} and score {nearest_person.scoreMatching}")
                        nearest_person.scoreMatching = min(
                            nearest_person.scoreMatching * 2, 1)
                        return [nearest_person], nearest_person
                    else:
                        print(
                            f"Centroid person {nearest_person.fullname} below threshold {threshold} and centroid got {centroid_similarity}")
                        return [], None

            # Return only the best match if it meets threshold for normal persons
            if nearest_person and nearest_person.scoreMatching >= threshold:
                nearest_person.scoreMatching = min(
                    nearest_person.scoreMatching * 2, 1)
                print(
                    f"Found best match: {nearest_person.fullname} with score {nearest_person.scoreMatching:.3f}")
                return [nearest_person], nearest_person
            if nearest_person and nearest_person.scoreMatching >= similarity_threshold:
                print(
                    f"Nearest match below request threshold: {nearest_person.fullname} with raw score {nearest_person.scoreMatching:.3f}, threshold {threshold}")
            print(f"No person found above threshold {threshold}")
            return [], None

        except Exception as e:
            print(f"Error in vectorized search: {e}")
            # Fallback to original method if vectorized search fails
            return await self._search_person_fallback(face_embedding, threshold)

    async def _search_person_fallback(self, face_embedding: np.ndarray, threshold: float = 0.5) -> Tuple[List[FaceResult], Optional[FaceResult]]:
        """
        Fallback search method using original loop-based approach.
        """
        results: List[FaceResult] = []
        nearest_person = None
        for person in self.persons:
            try:
                face_feature = person.get('FaceFeature')
                similarity = cosine_similarity(face_embedding, face_feature)
                results.append(FaceResult(
                    personId=person['personId'],
                    personCode=person['personCode'],
                    fullname=person['fullname'],
                    department_id=str(person.get('deptName') or person.get('department_id') or ''),
                    person_type=_get_person_type(person),
                    scoreMatching=float(similarity)
                ))
            except Exception as e:
                print(f"Error processing person {person['personId']}: {e}")
                continue

        # Sort by similarity in descending order
        results.sort(key=lambda x: x.scoreMatching, reverse=True)
        if not results:
            print("No results found")
            return [], None

        nearest_person = results[0]

        # Check if the best match is a hard to recognize person
        if nearest_person.personCode in self.hard_to_recognize_codes:
            print(
                f"Hard to recognize person detected (fallback): {nearest_person.fullname} ({nearest_person.personCode})")
            # Use fixed threshold of 0.7 for hard to recognize persons
            hard_threshold = 0.35
            if nearest_person.scoreMatching >= hard_threshold:
                nearest_person.scoreMatching = min(
                    nearest_person.scoreMatching * 2, 1)
                print(
                    f"Found hard to recognize person (fallback): {nearest_person.fullname} with score {nearest_person.scoreMatching:.3f}")
                return [nearest_person], nearest_person
            else:
                print(
                    f"Hard to recognize person {nearest_person.fullname} below threshold {hard_threshold}")
                return [], None

        # Normal person processing
        if nearest_person.scoreMatching >= threshold:
            nearest_person.scoreMatching = min(
                nearest_person.scoreMatching * 2, 1)
            return [nearest_person], nearest_person

        return [], None

    def clear_cache(self):
        """Clear the face features matrix to free memory."""
        if self.use_gpu and self.face_features_matrix is not None:
            del self.face_features_matrix
            torch.cuda.empty_cache()
        self.face_features_matrix = None
        self.person_indices = []
        print("Cleared face features cache")

    def get_embedding_dimension(self) -> int:
        """Get the dimension of face embeddings."""
        if self.face_features_matrix is not None:
            return self.face_features_matrix.shape[1]
        return 0

    def validate_embedding(self, embedding: np.ndarray) -> bool:
        """Validate if embedding has correct dimension (512)."""
        if embedding is None:
            return False

        actual_dim = embedding.shape[0] if embedding.ndim == 1 else embedding.size
        expected_dim = 512

        if actual_dim != expected_dim:
            print(
                f"Warning: Embedding dimension mismatch. Expected: {expected_dim}, Got: {actual_dim}")
            return False
        return True


# Create a singleton instance
database_manager = DatabaseManager()
