"""Face matching and recognition module with Faiss support."""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
from loguru import logger

from frs.database.models import Database, Identity
from frs.utils.config import config


class FaceMatcher:
    """Face matcher using cosine similarity or L2 distance with Faiss acceleration."""

    def __init__(self, db: Database):
        """Initialize face matcher.

        Args:
            db: Database instance for gallery management
        """
        self.db = db
        self.metric = config.matching.metric
        self.threshold = config.matching.threshold
        self.top_k = config.matching.top_k
        self.min_confidence = config.matching.min_confidence
        self.use_faiss = config.matching.use_faiss
        
        # Faiss index
        self.index = None
        self.identity_map = {}  # Maps index position to identity_id
        
        if self.use_faiss:
            self._initialize_faiss_index()

    def _initialize_faiss_index(self):
        """Initialize Faiss index."""
        embedding_dim = config.embedding.embedding_size
        
        if self.metric == "cosine":
            # Inner product for cosine similarity (embeddings are L2 normalized)
            self.index = faiss.IndexFlatIP(embedding_dim)
        elif self.metric == "l2":
            # L2 distance
            self.index = faiss.IndexFlatL2(embedding_dim)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
        
        logger.info(f"Initialized Faiss index with metric={self.metric}")

    def add_identity(
        self,
        identity_id: str,
        name: str,
        embedding: np.ndarray,
        image_path: str = None,
        metadata: dict = None
    ) -> bool:
        """Add identity to gallery.

        Args:
            identity_id: Unique identity identifier
            name: Person name
            embedding: Face embedding vector
            image_path: Path to reference image
            metadata: Additional metadata (JSON)

        Returns:
            True if added successfully
        """
        try:
            session = self.db.get_session()
            
            # Check if identity exists
            existing = session.query(Identity).filter_by(identity_id=identity_id).first()
            if existing:
                logger.warning(f"Identity {identity_id} already exists, updating...")
                existing.name = name
                existing.embedding = embedding.tobytes()
                existing.image_path = image_path
                existing.meta_data = json.dumps(metadata) if metadata else None
            else:
                # Add new identity
                identity = Identity(
                    identity_id=identity_id,
                    name=name,
                    embedding=embedding.tobytes(),
                    image_path=image_path,
                    meta_data=json.dumps(metadata) if metadata else None
                )
                session.add(identity)
            
            session.commit()
            session.close()
            
            # Update Faiss index
            if self.use_faiss:
                self._rebuild_faiss_index()
            
            logger.info(f"Added identity: {identity_id} ({name})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add identity: {e}")
            return False

    def remove_identity(self, identity_id: str) -> bool:
        """Remove identity from gallery.

        Args:
            identity_id: Identity identifier to remove

        Returns:
            True if removed successfully
        """
        try:
            session = self.db.get_session()
            identity = session.query(Identity).filter_by(identity_id=identity_id).first()
            
            if identity:
                session.delete(identity)
                session.commit()
                session.close()
                
                # Rebuild Faiss index
                if self.use_faiss:
                    self._rebuild_faiss_index()
                
                logger.info(f"Removed identity: {identity_id}")
                return True
            else:
                logger.warning(f"Identity not found: {identity_id}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to remove identity: {e}")
            return False

    def get_all_identities(self) -> List[Dict]:
        """Get all identities in gallery.

        Returns:
            List of identity dictionaries
        """
        session = self.db.get_session()
        identities = session.query(Identity).all()
        session.close()
        
        result = []
        for identity in identities:
            result.append({
                'identity_id': identity.identity_id,
                'name': identity.name,
                'image_path': identity.image_path,
                'created_at': identity.created_at.isoformat(),
                'metadata': json.loads(identity.meta_data) if identity.meta_data else None
            })
        
        return result

    def _rebuild_faiss_index(self):
        """Rebuild Faiss index from database."""
        if not self.use_faiss:
            return
        
        session = self.db.get_session()
        identities = session.query(Identity).all()
        session.close()
        
        if len(identities) == 0:
            logger.warning("No identities in database, index is empty")
            return
        
        # Reset index
        self._initialize_faiss_index()
        self.identity_map = {}
        
        # Add all embeddings to index
        embeddings = []
        for idx, identity in enumerate(identities):
            emb = np.frombuffer(identity.embedding, dtype=np.float32)
            embeddings.append(emb)
            self.identity_map[idx] = identity.identity_id
        
        embeddings = np.vstack(embeddings)
        
        # Normalize for cosine similarity
        if self.metric == "cosine":
            faiss.normalize_L2(embeddings)
        
        self.index.add(embeddings)
        logger.info(f"Rebuilt Faiss index with {len(identities)} identities")

    def match(self, embedding: np.ndarray) -> List[Dict]:
        """Match embedding against gallery.

        Args:
            embedding: Query embedding vector

        Returns:
            List of matches sorted by confidence, each containing:
                - identity_id: Matched identity ID
                - name: Person name
                - confidence: Match confidence (0-1)
                - distance: Distance metric value
        """
        if self.use_faiss and self.index and self.index.ntotal > 0:
            return self._match_faiss(embedding)
        else:
            return self._match_database(embedding)

    def _match_faiss(self, embedding: np.ndarray) -> List[Dict]:
        """Match using Faiss index.

        Args:
            embedding: Query embedding vector

        Returns:
            List of top-K matches
        """
        # Reshape and normalize
        query = embedding.reshape(1, -1).astype(np.float32)
        if self.metric == "cosine":
            faiss.normalize_L2(query)
        
        # Search
        k = min(self.top_k, self.index.ntotal)
        distances, indices = self.index.search(query, k)
        
        # Get identity info from database
        session = self.db.get_session()
        matches = []
        
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:  # No match found
                continue
            
            identity_id = self.identity_map[idx]
            identity = session.query(Identity).filter_by(identity_id=identity_id).first()
            
            if not identity:
                continue
            
            # Convert distance to confidence
            if self.metric == "cosine":
                # Inner product (higher is better, range: -1 to 1)
                confidence = (dist + 1) / 2  # Normalize to 0-1
            else:
                # L2 distance (lower is better)
                confidence = 1 / (1 + dist)  # Convert to similarity
            
            # Apply threshold
            if confidence < self.min_confidence:
                continue
            
            matches.append({
                'identity_id': identity.identity_id,
                'name': identity.name,
                'confidence': float(confidence),
                'distance': float(dist),
                'image_path': identity.image_path
            })
        
        session.close()
        
        # Sort by confidence
        matches.sort(key=lambda x: x['confidence'], reverse=True)
        return matches

    def _match_database(self, embedding: np.ndarray) -> List[Dict]:
        """Match using database scan (fallback without Faiss).

        Args:
            embedding: Query embedding vector

        Returns:
            List of top-K matches
        """
        session = self.db.get_session()
        identities = session.query(Identity).all()
        session.close()
        
        if len(identities) == 0:
            return []
        
        # Compute similarities
        matches = []
        for identity in identities:
            gallery_emb = np.frombuffer(identity.embedding, dtype=np.float32)
            
            if self.metric == "cosine":
                # Cosine similarity
                similarity = np.dot(embedding, gallery_emb)
            else:
                # L2 distance
                dist = np.linalg.norm(embedding - gallery_emb)
                similarity = 1 / (1 + dist)
            
            # Apply threshold
            if similarity < self.min_confidence:
                continue
            
            matches.append({
                'identity_id': identity.identity_id,
                'name': identity.name,
                'confidence': float(similarity),
                'distance': float(1 - similarity),
                'image_path': identity.image_path
            })
        
        # Sort by confidence and return top-K
        matches.sort(key=lambda x: x['confidence'], reverse=True)
        return matches[:self.top_k]

    def batch_match(self, embeddings: np.ndarray) -> List[List[Dict]]:
        """Match multiple embeddings.

        Args:
            embeddings: Array of embeddings (N, embedding_size)

        Returns:
            List of match lists for each query
        """
        if self.use_faiss and self.index and self.index.ntotal > 0:
            return self._batch_match_faiss(embeddings)
        else:
            return [self.match(emb) for emb in embeddings]

    def _batch_match_faiss(self, embeddings: np.ndarray) -> List[List[Dict]]:
        """Batch match using Faiss.

        Args:
            embeddings: Array of embeddings (N, embedding_size)

        Returns:
            List of match lists
        """
        # Normalize if using cosine similarity
        queries = embeddings.astype(np.float32)
        if self.metric == "cosine":
            faiss.normalize_L2(queries)
        
        # Search
        k = min(self.top_k, self.index.ntotal)
        distances, indices = self.index.search(queries, k)
        
        # Get identity info
        session = self.db.get_session()
        batch_matches = []
        
        for dists, idxs in zip(distances, indices):
            matches = []
            for dist, idx in zip(dists, idxs):
                if idx == -1:
                    continue
                
                identity_id = self.identity_map[idx]
                identity = session.query(Identity).filter_by(identity_id=identity_id).first()
                
                if not identity:
                    continue
                
                # Convert distance to confidence
                if self.metric == "cosine":
                    confidence = (dist + 1) / 2
                else:
                    confidence = 1 / (1 + dist)
                
                if confidence < self.min_confidence:
                    continue
                
                matches.append({
                    'identity_id': identity.identity_id,
                    'name': identity.name,
                    'confidence': float(confidence),
                    'distance': float(dist)
                })
            
            matches.sort(key=lambda x: x['confidence'], reverse=True)
            batch_matches.append(matches)
        
        session.close()
        return batch_matches

    def save_index(self, path: str):
        """Save Faiss index and identity map to disk.

        Args:
            path: Directory path to save index
        """
        if not self.use_faiss or not self.index:
            logger.warning("No Faiss index to save")
            return
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save Faiss index
        faiss.write_index(self.index, str(path / "faiss.index"))
        
        # Save identity map
        with open(path / "identity_map.pkl", "wb") as f:
            pickle.dump(self.identity_map, f)
        
        logger.info(f"Saved Faiss index to {path}")

    def load_index(self, path: str):
        """Load Faiss index and identity map from disk.

        Args:
            path: Directory path containing saved index
        """
        if not self.use_faiss:
            return
        
        path = Path(path)
        
        # Load Faiss index
        index_file = path / "faiss.index"
        if index_file.exists():
            self.index = faiss.read_index(str(index_file))
            logger.info(f"Loaded Faiss index from {index_file}")
        
        # Load identity map
        map_file = path / "identity_map.pkl"
        if map_file.exists():
            with open(map_file, "rb") as f:
                self.identity_map = pickle.load(f)
            logger.info(f"Loaded identity map with {len(self.identity_map)} entries")
