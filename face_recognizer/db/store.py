"""SQLite + FAISS identity store with row-ID linking.

SQLite is the source of truth.  The FAISS ``IndexFlatIP`` is rebuilt from
SQLite rows on ``load()`` and after any ``delete()``.  FAISS index
positions are maintained in lockstep with the order of iteration over
SQLite rows via the ``_faiss_to_row`` mapping list.

Usage::

    store = IdentityStore(db_path="data/identity.db", index_path="data/faiss.index")
    store.load()                          # rebuild FAISS from existing SQLite rows
    row_id = store.enroll("Alice", embedding, photo_hash="abc123", quality_score=0.92)
    results = store.search(query_embedding, k=1)
    store.list_names()
    store.delete("Alice")
    store.save()
"""

from __future__ import annotations

import logging
import os
import sqlite3
import struct
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# Prevent macOS OpenMP runtime conflicts when PyTorch and FAISS both
# try to load the same shared library.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

logger = logging.getLogger(__name__)

# Embedding dimension (ResNet-100 AdaFace / ArcFace).
_DIM = 512
# Binary size of one embedding: 512 float32 × 4 bytes = 2048.
_BLOB_FMT = f"{_DIM}f"
_BLOB_SIZE = _DIM * 4

# ---------------------------------------------------------------------------
# SQL schema
# ---------------------------------------------------------------------------

_CREATE_TABLE_SQL = """\
CREATE TABLE IF NOT EXISTS identities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    embedding BLOB NOT NULL,
    photo_hash TEXT,
    quality_score REAL,
    enrolled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

_CREATE_INDEX_SQL = "CREATE INDEX IF NOT EXISTS idx_name ON identities(name);"


# ===================================================================
# IdentityStore
# ===================================================================


class IdentityStore:
    """Persistent identity store backed by SQLite + FAISS.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database file.
    index_path : str
        Path for persisting the FAISS index (secondary; SQLite is canonical).
    threshold : float
        Cosine-similarity threshold for a positive match (0.0 – 1.0).
    """

    def __init__(
        self,
        *,
        db_path: str = "data/identity.db",
        index_path: str = "data/faiss.index",
        threshold: float = 0.65,
    ) -> None:
        self.db_path = str(db_path)
        self.index_path = str(index_path)
        self.threshold = threshold

        # Lazily initialised internals.
        self._conn: Optional[sqlite3.Connection] = None
        self._index = None  # faiss.IndexFlatIP
        # Maps FAISS internal position → SQLite ``identities.id``.
        self._faiss_to_row: List[int] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_conn(self) -> sqlite3.Connection:
        """Return (and maybe create) the SQLite connection and schema."""
        if self._conn is None:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.db_path)
            self._create_tables()
        return self._conn

    def _ensure_index(self):
        """Return the FAISS index, creating it lazily to avoid import
        conflicts with PyTorch."""
        if self._index is None:
            import faiss  # noqa: PLC0415  -- deferred import

            self._index = faiss.IndexFlatIP(_DIM)
        return self._index

    def _create_tables(self) -> None:
        """Ensure the ``identities`` table and index exist."""
        conn = self._ensure_conn()
        conn.execute(_CREATE_TABLE_SQL)
        conn.execute(_CREATE_INDEX_SQL)
        conn.commit()

    @staticmethod
    def _pack_embedding(embedding: np.ndarray) -> bytes:
        """Convert a (512,) float32 vector to a 2048-byte BLOB."""
        emb = np.asarray(embedding, dtype=np.float32).ravel()
        if emb.shape[0] != _DIM:
            raise ValueError(
                f"Embedding must be {_DIM}-dimensional, got {emb.shape[0]}"
            )
        return struct.pack(_BLOB_FMT, *emb)

    @staticmethod
    def _unpack_embedding(blob: bytes) -> np.ndarray:
        """Convert a 2048-byte BLOB back to a (512,) float32 vector."""
        return np.array(struct.unpack(_BLOB_FMT, blob), dtype=np.float32)

    # ------------------------------------------------------------------
    # Persistence (SQLite is canonical; FAISS is secondary)
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Rebuild the FAISS index from all rows in the SQLite database.

        This is the canonical load path — SQLite is always the source of
        truth.  If the database file does not exist yet it will be created
        (empty), resulting in an empty index.

        Safe to call multiple times; each call fully rebuilds state.
        """
        import faiss  # noqa: PLC0415

        conn = self._ensure_conn()

        # Fresh FAISS index.
        self._index = faiss.IndexFlatIP(_DIM)
        self._faiss_to_row = []

        rows = conn.execute(
            "SELECT id, embedding FROM identities ORDER BY id"
        ).fetchall()

        for row_id, blob in rows:
            emb = self._unpack_embedding(blob)
            # FAISS expects (1, D) shape for a single vector.
            self._index.add(emb.reshape(1, -1))
            self._faiss_to_row.append(row_id)

        logger.info(
            "Loaded %d identity rows from %s", len(rows), self.db_path
        )

    def save(self) -> None:
        """Persist the current FAISS index to *index_path*.

        The SQLite database is saved automatically on every
        insert / delete via ``COMMIT``, so this method only writes the
        FAISS file.
        """
        if self._index is None:
            return
        import faiss  # noqa: PLC0415

        path = Path(self.index_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path))
        logger.debug("Saved FAISS index (%d vectors) to %s", self._index.ntotal, path)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def enroll(
        self,
        name: str,
        embedding: np.ndarray,
        photo_hash: Optional[str] = None,
        quality_score: Optional[float] = None,
    ) -> int:
        """Insert a new identity and add its vector to the FAISS index.

        Parameters
        ----------
        name : str
            Person identifier.  Duplicate names are allowed (multiple
            embedding rows per name) but a warning is emitted.
        embedding : np.ndarray
            A (512,) float32 embedding vector.  Will be normalised to
            unit length before storage.
        photo_hash : str or None
            SHA-256 hex digest of the enrollment photo.
        quality_score : float or None
            Face-quality score at enrollment time.

        Returns
        -------
        int
            The ``identities.id`` (SQLite row ID) of the new row.
        """
        conn = self._ensure_conn()

        # Normalise to unit length (cosine similarity via inner product).
        emb = np.asarray(embedding, dtype=np.float32).ravel()
        norm = np.linalg.norm(emb)
        if norm < 1e-12:
            raise ValueError("Embedding has near-zero norm; cannot normalise.")
        emb = emb / norm

        # Warn on duplicate name (allowed — multiple vectors per identity).
        existing = conn.execute(
            "SELECT COUNT(*) FROM identities WHERE name=?", (name,)
        ).fetchone()[0]
        if existing > 0:
            logger.warning(
                "Name %r already has %d row(s) — adding another vector.", name, existing
            )

        blob = self._pack_embedding(emb)

        cursor = conn.execute(
            "INSERT INTO identities (name, embedding, photo_hash, quality_score) "
            "VALUES (?, ?, ?, ?)",
            (name, blob, photo_hash, quality_score),
        )
        row_id = cursor.lastrowid
        conn.commit()

        # Append to FAISS at the next sequential position.
        self._ensure_index().add(emb.reshape(1, -1))
        self._faiss_to_row.append(row_id)

        self.save()

        logger.info(
            "Enrolled %r (row_id=%d, total_vectors=%d)",
            name,
            row_id,
            self._index.ntotal,
        )
        return row_id

    def search(
        self, embedding: np.ndarray, k: int = 1
    ) -> List[Tuple[str, float]]:
        """Find the *k* closest enrolled identities.

        Parameters
        ----------
        embedding : np.ndarray
            Query embedding, shape ``(512,)`` float32.
        k : int
            Maximum number of results to return.

        Returns
        -------
        list of (name, score)
            Tuples sorted by descending cosine similarity.  Only results
            whose score meets ``self.threshold`` are included.  Returns an
            empty list when the index is empty.
        """
        idx = self._ensure_index()
        if idx.ntotal == 0:
            return []

        # Normalise query.
        vec = np.asarray(embedding, dtype=np.float32).ravel()
        norm = np.linalg.norm(vec)
        if norm < 1e-12:
            return []
        vec = vec / norm

        k = min(k, idx.ntotal)
        scores, indices = idx.search(vec.reshape(1, -1), k)

        conn = self._ensure_conn()
        results: List[Tuple[str, float]] = []
        for faiss_pos, score in zip(indices[0], scores[0]):
            if faiss_pos < 0:  # FAISS padding sentinel.
                continue
            if float(score) < self.threshold:
                continue
            row_id = self._faiss_to_row[faiss_pos]
            row = conn.execute(
                "SELECT name FROM identities WHERE id=?", (row_id,)
            ).fetchone()
            if row is not None:
                results.append((row[0], float(score)))

        return results

    def enroll_many(
        self,
        name: str,
        embeddings: np.ndarray,
        photo_hash: Optional[str] = None,
        quality_scores: Optional[list[float]] = None,
    ) -> list[int]:
        """Insert multiple embedding rows for a single identity.

        Parameters
        ----------
        name : str
            Person identifier.
        embeddings : np.ndarray
            ``(N, 512)`` float32 matrix.  Each row is independently
            L2-normalised before storage.
        photo_hash : str or None
            SHA-256 hex digest of the original enrollment photo (shared
            across all rows).
        quality_scores : list of float or None
            Per-embedding quality scores; must match *N*.

        Returns
        -------
        list of int
            SQLite row IDs for each inserted vector.
        """
        conn = self._ensure_conn()
        emb = np.asarray(embeddings, dtype=np.float32)
        if emb.ndim == 1:
            emb = emb[np.newaxis, :]
        N = emb.shape[0]

        # Normalise each row individually.
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        emb = emb / norms

        row_ids: list[int] = []
        for i in range(N):
            blob = self._pack_embedding(emb[i])
            qs = float(quality_scores[i]) if quality_scores else None
            cursor = conn.execute(
                "INSERT INTO identities (name, embedding, photo_hash, quality_score) "
                "VALUES (?, ?, ?, ?)",
                (name, blob, photo_hash, qs),
            )
            row_ids.append(cursor.lastrowid)
            self._ensure_index().add(emb[i].reshape(1, -1))
            self._faiss_to_row.append(cursor.lastrowid)

        conn.commit()
        self.save()

        logger.info(
            "Enrolled %d vectors for %r (total=%d)",
            N, name, self._index.ntotal if self._index else 0,
        )
        return row_ids

    def delete(self, name: str) -> None:
        """Remove every row belonging to *name*.

        Because ``IndexFlatIP`` does not support single-vector removal,
        the FAISS index is rebuilt from the remaining SQLite rows after
        the ``DELETE``.
        """
        conn = self._ensure_conn()
        cursor = conn.execute("DELETE FROM identities WHERE name=?", (name,))
        deleted = cursor.rowcount
        conn.commit()

        if deleted > 0:
            # Rebuild FAISS from the remaining rows.
            self.load()
            self.save()
            logger.info(
                "Deleted %d row(s) for %r — FAISS index rebuilt.", deleted, name
            )

    def list_names(self) -> List[str]:
        """Return distinct enrolled names, sorted alphabetically."""
        conn = self._ensure_conn()
        rows = conn.execute(
            "SELECT DISTINCT name FROM identities ORDER BY name"
        ).fetchall()
        return [r[0] for r in rows]

    def calibrate_threshold(self, margin: float = 0.05) -> float:
        """Compute an adaptive threshold from the enrolled population.

        Finds the maximum cosine similarity between any pair of vectors
        belonging to *different* enrolled identities, then adds *margin*.
        This reflects the actual FAISS search condition — each query is
        compared against every stored vector, not just centroids.

        Parameters
        ----------
        margin : float
            Buffer added to max cross-identity similarity (default 0.05).

        Returns
        -------
        float
            Recommended cosine-similarity threshold in (0, 1).  Returns 0.0
            when fewer than 2 identities are enrolled.
        """
        conn = self._ensure_conn()
        rows = conn.execute(
            "SELECT name, embedding FROM identities ORDER BY id"
        ).fetchall()

        if len(rows) < 2:
            logger.info("Fewer than 2 identities — cannot calibrate threshold.")
            return 0.0

        names = [r[0] for r in rows]
        embs = np.stack([self._unpack_embedding(r[1]) for r in rows], axis=0)

        # Build a mask that excludes same-identity pairs
        name_arr = np.array(names)
        same_id = name_arr[:, None] == name_arr[None, :]  # (N, N) bool

        # Cosine similarity matrix
        sims = embs @ embs.T
        # Exclude self-pairs AND same-identity pairs
        mask = ~np.eye(len(sims), dtype=bool) & ~same_id

        if not mask.any():
            logger.info("Only one identity with vectors — cannot calibrate.")
            return 0.0

        max_inter = float(sims[mask].max())
        idx_flat = np.argmax(sims * mask)
        i, j = idx_flat // len(sims), idx_flat % len(sims)

        threshold = max_inter + margin
        logger.info(
            "Calibrated threshold=%.4f (max cross-id=%.4f: %r ↔ %r, margin=%.2f, "
            "%d identities, %d vectors)",
            threshold, max_inter, names[i], names[j], margin,
            len(set(names)), len(rows),
        )
        self.threshold = threshold
        return threshold
