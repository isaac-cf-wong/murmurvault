"""Tests for chunking, fusion and the hybrid search index."""

from __future__ import annotations

from itertools import pairwise
from typing import ClassVar

import numpy as np
import pytest

from murmurvault.search import Index, chunk_segments, fts_query, rrf
from murmurvault.vault import Segment, Transcript, Vault


@pytest.fixture
def vault(isolated_env):
    """Provide an empty vault at the isolated location.

    Args:
        isolated_env: Fixture that isolates config and vault paths.

    Returns:
        The vault.
    """
    return Vault(isolated_env["vault"])


def test_chunking_overlaps_and_covers_everything():
    """Chunks cover the whole transcript and overlap by one segment."""
    segs = [Segment(i, i + 1, "word " * 30, speaker="me") for i in range(10)]
    chunks = chunk_segments(segs, max_chars=300)
    assert chunks[0]["start"] == 0
    assert chunks[-1]["end"] == 10
    assert all(c["speaker"] == "me" for c in chunks)
    for a, b in pairwise(chunks):
        assert b["start"] <= a["end"]  # one-segment overlap


def test_rrf_and_fts_query():
    """RRF favours items ranked high in several lists; FTS queries quote every word."""
    scores = rrf([[1, 2, 3], [3, 1]])
    assert max(scores, key=scores.get) == 1
    assert fts_query('budget "Q3" OR-not') == '"budget" OR "q3" OR "or" OR "not"'
    assert not fts_query("!!!")


def test_keyword_search_with_filters(vault, isolated_env):
    """Keyword search ranks by BM25 and honours folder and tag filters."""
    a = vault.new_recording(title="Budget", folder="work", tags=["finance"])
    b = vault.new_recording(title="Trip", folder="home")
    vault.save_transcript(a, "x", Transcript("e", "m", [Segment(0, 3, "we cut the travel budget", "me")]))
    vault.save_transcript(b, "x", Transcript("e", "m", [Segment(5, 8, "travel plans for the summer")]))
    index = Index(vault, isolated_env)
    assert index.rebuild() == (2, 2)
    hits = index.search("travel budget")
    assert [h.rec_id for h in hits] == [a.id, b.id]
    assert [h.rec_id for h in index.search("travel", folder="home")] == [b.id]
    assert [h.rec_id for h in index.search("travel", tags=["finance"])] == [a.id]
    index.remove(a.id)
    assert [h.rec_id for h in index.search("budget")] == []
    index.close()


class FakeEmbedder:
    """Bag-of-words embedder in which money, budget and cost are synonyms."""

    enabled = True
    model_id = "fake:1"
    VOCAB: ClassVar[list[str]] = ["money", "budget", "cost", "holiday", "trip"]

    def embed(self, texts, query=False):
        """Embed texts.

        Args:
            texts: Texts to embed.
            query: Ignored.

        Returns:
            Unit vectors.
        """
        out = []
        for t in texts:
            v = np.array([t.lower().count(w) for w in self.VOCAB], np.float32)
            v[0] = v[1] = v[2] = v[:3].sum()  # synonyms share a direction
            out.append(v / max(np.linalg.norm(v), 1e-9))
        return np.stack(out)


def test_semantic_search_finds_synonyms_and_detects_model_change(vault, isolated_env, caplog):
    """Vector search finds synonyms; a changed embedding model falls back to keywords and says so."""
    a = vault.new_recording(title="Costs")
    vault.save_transcript(a, "x", Transcript("e", "m", [Segment(0, 3, "the cost is too high")]))
    index = Index(vault, isolated_env)
    index.embedder = FakeEmbedder()
    index.rebuild()
    assert [h.rec_id for h in index.search("money")] == [a.id]  # no keyword overlap
    index.embedder.model_id = "fake:2"
    assert index.search("money") == []
    assert "reindex" in caplog.text
    index.close()
