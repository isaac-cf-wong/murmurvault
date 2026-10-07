"""Derived search index: SQLite FTS5 (BM25) + embeddings, fused with reciprocal rank fusion, then reranked.

The index lives in ``<vault>/.murmurvault/index.db`` and can always be rebuilt from the vault.
Vectors are brute-forced with numpy, which is fast enough for tens of thousands of chunks.
"""

# sentence-transformers (optional `search` extra) and httpx are imported on first use.
# ruff: noqa: PLC0415

from __future__ import annotations

import logging
import re
import sqlite3
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from murmurvault import config as config_mod
from murmurvault.vault import Recording, Segment, Transcript, Vault

logger = logging.getLogger(__name__)

CHUNK_CHARS = 600
RRF_K = 60

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    rec_id TEXT NOT NULL,
    start REAL, end REAL,
    speaker TEXT,
    text TEXT NOT NULL,
    emb BLOB
);
CREATE INDEX IF NOT EXISTS chunks_rec ON chunks(rec_id);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text, content='chunks', content_rowid='id', tokenize='porter unicode61'
);
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;
"""


@dataclass
class Hit:
    """A search result: one chunk of a transcript.

    Attributes:
        rec_id: Recording id.
        title: Recording title.
        folder: Recording folder.
        start: Chunk start time in seconds.
        end: Chunk end time in seconds.
        speaker: Comma-separated speakers in the chunk.
        text: Chunk text.
        score: Reranker score, or the fused rank score when reranking is off.
    """

    rec_id: str
    title: str
    folder: str
    start: float
    end: float
    speaker: str | None
    text: str
    score: float

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dict.

        Returns:
            The hit as plain data.
        """
        return asdict(self)


def chunk_segments(segments: list[Segment], max_chars: int = CHUNK_CHARS) -> list[dict[str, Any]]:
    """Group consecutive segments into windows of about ``max_chars``, overlapping by one segment.

    Args:
        segments: Transcript segments in time order.
        max_chars: Target window size in characters.

    Returns:
        Chunks with ``start``, ``end``, ``speaker`` and ``text``.
    """
    chunks: list[dict[str, Any]] = []
    window: list[Segment] = []

    def flush():
        speakers = list(dict.fromkeys(s.speaker for s in window if s.speaker))
        text = " ".join(f"{s.speaker}: {s.text}" if s.speaker else s.text for s in window)
        chunks.append(
            {"start": window[0].start, "end": window[-1].end, "speaker": ", ".join(speakers) or None, "text": text}
        )

    size = 0
    for seg in segments:
        if not seg.text.strip():
            continue
        window.append(seg)
        size += len(seg.text) + 1
        if size >= max_chars:
            flush()
            window = window[-1:]
            size = len(window[0].text) + 1
    if window and (not chunks or window[-1].end > chunks[-1]["end"]):
        flush()
    return chunks


def fts_query(text: str) -> str:
    """Turn free text into an FTS5 query that matches any of its words.

    Args:
        text: The user's query.

    Returns:
        Quoted terms joined by ``OR``; empty if the text has no words.
    """
    terms = re.findall(r"\w+", text.lower())
    return " OR ".join(f'"{t}"' for t in terms)


def rrf(rankings: list[list[int]], k: int = RRF_K) -> dict[int, float]:
    """Reciprocal rank fusion.

    Args:
        rankings: Ranked lists of item ids, best first.
        k: Damping constant.

    Returns:
        Item id to fused score.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank + 1)
    return scores


# -- embedding and reranking backends ---------------------------------------------------------------

_MODELS: dict[tuple[str, str], Any] = {}


def _auth(cfg: dict) -> dict[str, str]:
    key = config_mod.api_key(cfg)
    return {"Authorization": f"Bearer {key}"} if key else {}


class Embedder:
    """Text embeddings from a local sentence-transformers model or an OpenAI-compatible API."""

    def __init__(self, cfg: dict):
        """Configure from the ``[embedding]`` section.

        Args:
            cfg: The ``[embedding]`` config section.
        """
        self.cfg = cfg
        self.backend = cfg.get("backend", "local")
        self.model_id = f"{self.backend}:{cfg.get('model')}"

    @property
    def enabled(self) -> bool:
        """Whether embeddings are configured at all."""
        return self.backend != "none"

    def embed(self, texts: list[str], query: bool = False) -> np.ndarray:
        """Embed texts as unit vectors.

        Args:
            texts: Texts to embed.
            query: Prefix with the configured query instruction.

        Returns:
            A ``(len(texts), dim)`` float32 array of unit vectors.

        Raises:
            RuntimeError: If the local backend's extra is missing.
            ValueError: If the backend is unknown.
        """
        if query and self.cfg.get("query_prefix"):
            texts = [self.cfg["query_prefix"] + t for t in texts]
        if self.backend == "local":
            key = ("embed", self.cfg["model"])
            if key not in _MODELS:
                try:
                    from sentence_transformers import SentenceTransformer
                except ImportError as exc:
                    raise RuntimeError("local embeddings need `pip install 'murmurvault[search]'`") from exc
                _MODELS[key] = SentenceTransformer(self.cfg["model"])
            vecs = _MODELS[key].encode(texts, normalize_embeddings=True, convert_to_numpy=True)
            return np.asarray(vecs, dtype=np.float32)
        if self.backend == "openai":
            import httpx

            resp = httpx.post(
                f"{self.cfg['base_url'].rstrip('/')}/embeddings",
                headers=_auth(self.cfg),
                json={"model": self.cfg["model"], "input": texts},
                timeout=120,
            )
            resp.raise_for_status()
            data = sorted(resp.json()["data"], key=lambda d: d["index"])
            vecs = np.asarray([d["embedding"] for d in data], dtype=np.float32)
            return vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12)
        raise ValueError(f"unknown embedding backend {self.backend!r}")


class Reranker:
    """Query-document relevance from a local cross-encoder or a ``/v1/rerank`` API."""

    def __init__(self, cfg: dict):
        """Configure from the ``[rerank]`` section.

        Args:
            cfg: The ``[rerank]`` config section.
        """
        self.cfg = cfg
        self.backend = cfg.get("backend", "local")

    @property
    def enabled(self) -> bool:
        """Whether reranking is configured at all."""
        return self.backend != "none"

    def scores(self, query: str, docs: list[str]) -> list[float]:
        """Score each document's relevance to the query.

        Args:
            query: The query.
            docs: Candidate documents.

        Returns:
            One score per document; higher is more relevant.

        Raises:
            RuntimeError: If the local backend's extra is missing.
            ValueError: If the backend is unknown.
        """
        if self.backend == "local":
            key = ("rerank", self.cfg["model"])
            if key not in _MODELS:
                try:
                    from sentence_transformers import CrossEncoder
                except ImportError as exc:
                    raise RuntimeError("local reranking needs `pip install 'murmurvault[search]'`") from exc
                _MODELS[key] = CrossEncoder(self.cfg["model"])
            return [float(s) for s in _MODELS[key].predict([(query, d) for d in docs])]
        if self.backend == "api":
            import httpx

            resp = httpx.post(
                f"{self.cfg['base_url'].rstrip('/')}/rerank",
                headers=_auth(self.cfg),
                json={"model": self.cfg["model"], "query": query, "documents": docs, "top_n": len(docs)},
                timeout=120,
            )
            resp.raise_for_status()
            out = [0.0] * len(docs)
            for r in resp.json()["results"]:
                out[r["index"]] = float(r.get("relevance_score", r.get("score", 0.0)))
            return out
        raise ValueError(f"unknown rerank backend {self.backend!r}")


# -- the index ------------------------------------------------------------------------------------


class Index:
    """The search index of one vault."""

    def __init__(self, vault: Vault, cfg: dict):
        """Open (and create if needed) the index database.

        Args:
            vault: The vault to index.
            cfg: The configuration.
        """
        self.vault = vault
        self.db = sqlite3.connect(vault.internal / "index.db")
        self.db.executescript(SCHEMA)
        self.embedder = Embedder(cfg["embedding"])
        self.reranker = Reranker(cfg["rerank"])

    def close(self) -> None:
        """Close the database."""
        self.db.close()

    def _meta(self, key: str) -> str | None:
        row = self.db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def _set_meta(self, key: str, value: str) -> None:
        self.db.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", (key, value))

    def remove(self, rec_id: str) -> None:
        """Drop a recording from the index.

        Args:
            rec_id: Recording id.
        """
        self.db.execute("DELETE FROM chunks WHERE rec_id = ?", (rec_id,))
        self.db.commit()

    def update(self, rec: Recording, transcript: Transcript | None = None) -> int:
        """(Re-)index a recording's transcript.

        If embedding fails, the chunks are still indexed for keyword search.

        Args:
            rec: The recording.
            transcript: Transcript to index; defaults to the active one.

        Returns:
            Number of chunks indexed.
        """
        transcript = transcript or self.vault.load_transcript(rec)
        self.db.execute("DELETE FROM chunks WHERE rec_id = ?", (rec.id,))
        if transcript is None:
            self.db.commit()
            return 0
        chunks = chunk_segments(transcript.segments)
        vecs: np.ndarray | None = None
        if chunks and self.embedder.enabled:
            try:
                vecs = self.embedder.embed([c["text"] for c in chunks])
                self._set_meta("embedding_model", self.embedder.model_id)
            except Exception as exc:  # noqa: BLE001 -- missing model or API outage must not block indexing
                logger.warning("embeddings skipped (%s); keyword search still works", exc)
        for i, c in enumerate(chunks):
            emb = vecs[i].tobytes() if vecs is not None else None
            self.db.execute(
                "INSERT INTO chunks(rec_id, start, end, speaker, text, emb) VALUES (?, ?, ?, ?, ?, ?)",
                (rec.id, c["start"], c["end"], c["speaker"], c["text"], emb),
            )
        self.db.commit()
        return len(chunks)

    def rebuild(self) -> tuple[int, int]:
        """Rebuild the whole index from the vault.

        Returns:
            Number of recordings and number of chunks indexed.
        """
        self.db.execute("DELETE FROM chunks")
        self.db.execute("DELETE FROM meta")
        self.db.commit()
        recs = self.vault.recordings()
        n = sum(self.update(r) for r in recs)
        return len(recs), n

    def _keyword(self, query: str, allowed: set[str], limit: int) -> list[int]:
        q = fts_query(query)
        if not q:
            return []
        rows = self.db.execute(
            "SELECT c.id, c.rec_id FROM chunks_fts f JOIN chunks c ON c.id = f.rowid "
            "WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts) LIMIT ?",
            (q, limit * 4),
        ).fetchall()
        return [cid for cid, rid in rows if rid in allowed][:limit]

    def _semantic(self, query: str, allowed: set[str], limit: int) -> list[int]:
        if not self.embedder.enabled:
            return []
        stored = self._meta("embedding_model")
        if not stored:
            return []
        if stored != self.embedder.model_id:
            logger.warning(
                "index was embedded with %s, config says %s; run `murmurvault reindex`. Using keyword search only.",
                stored,
                self.embedder.model_id,
            )
            return []
        rows = [
            r
            for r in self.db.execute("SELECT id, rec_id, emb FROM chunks WHERE emb IS NOT NULL").fetchall()
            if r[1] in allowed
        ]
        if not rows:
            return []
        try:
            qv = self.embedder.embed([query], query=True)[0]
        except Exception as exc:  # noqa: BLE001 -- fall back to keyword search
            logger.warning("semantic search skipped (%s)", exc)
            return []
        mat = np.stack([np.frombuffer(r[2], dtype=np.float32) for r in rows])
        order = np.argsort(-(mat @ qv))[:limit]
        return [rows[i][0] for i in order]

    def search(  # noqa: PLR0913
        self,
        query: str,
        *,
        k: int = 10,
        folder: str | None = None,
        tags: list[str] | None = None,
        rerank: bool = True,
        candidates: int = 50,
    ) -> list[Hit]:
        """Hybrid search: BM25 and vector candidates fused by RRF, then reranked.

        Args:
            query: Free-text query.
            k: Number of hits to return.
            folder: Only recordings in this folder or below.
            tags: Only recordings carrying all these tags.
            rerank: Rerank the fused candidates.
            candidates: Candidates taken from each retriever.

        Returns:
            The best hits, best first.
        """
        recs = {r.id: r for r in self.vault.recordings(folder=folder, tags=tags)}
        if not recs:
            return []
        allowed = set(recs)
        fused = rrf([self._keyword(query, allowed, candidates), self._semantic(query, allowed, candidates)])
        ranked = sorted(fused, key=fused.__getitem__, reverse=True)[: max(k * 3, 20)]
        if not ranked:
            return []
        placeholders = ",".join("?" * len(ranked))
        rows = self.db.execute(
            # Only "?" placeholders are interpolated; the values are bound parameters.
            f"SELECT id, rec_id, start, end, speaker, text FROM chunks WHERE id IN ({placeholders})",  # noqa: S608
            ranked,
        ).fetchall()
        by_id = {r[0]: r for r in rows}
        ranked = [cid for cid in ranked if cid in by_id]
        scores = {cid: fused[cid] for cid in ranked}

        if rerank and self.reranker.enabled and len(ranked) > 1:
            try:
                rs = self.reranker.scores(query, [by_id[c][5] for c in ranked])
                scores = dict(zip(ranked, rs, strict=True))
                ranked = sorted(ranked, key=scores.__getitem__, reverse=True)
            except Exception as exc:  # noqa: BLE001 -- keep the fused order
                logger.warning("reranking skipped (%s)", exc)

        hits = []
        for cid in ranked[:k]:
            _, rid, start, end, speaker, text = by_id[cid]
            rec = recs[rid]
            hits.append(Hit(rid, rec.title, rec.folder, start, end, speaker, text, round(scores[cid], 4)))
        return hits
