"""
Semantic Trigger Filter service.

Loads a sentence-transformers embedding model ONCE (the first time it's
needed, then cached), embeds the user's preference and each post's text,
scores every post with cosine similarity, and maps each score to an
action: SHOW, WARN, or FILTER.

This is genuine semantic matching, not keyword search: two sentences
with no words in common can still score a high similarity if their
*meaning* is related (e.g. "placement rejection" vs. "failed the
final interview").
"""

from functools import lru_cache
from typing import List

from sentence_transformers import SentenceTransformer, util

from app.config import settings
from app.schemas.semantic_schema import PostIn, SemanticResult


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    # lru_cache means this expensive load only happens once per process
    # (the first time get_model() is called), not on every request.
    return SentenceTransformer(settings.SEMANTIC_MODEL_NAME)


def _action_from_score(score: float) -> str:
    if score >= settings.SEMANTIC_FILTER_THRESHOLD:
        return "FILTER"
    if score >= settings.SEMANTIC_WARN_THRESHOLD:
        return "WARN"
    return "SHOW"


def score_posts(preference: str, posts: List[PostIn]) -> List[SemanticResult]:
    if not preference.strip():
        # No preference set yet -> nothing to filter against.
        return [
            SemanticResult(post_id=p.id, similarity=0.0, action="SHOW")
            for p in posts
        ]

    model = get_model()

    preference_embedding = model.encode(preference, convert_to_tensor=True)
    post_texts = [p.text for p in posts]
    post_embeddings = model.encode(post_texts, convert_to_tensor=True)

    similarities = util.cos_sim(preference_embedding, post_embeddings)[0]

    results: List[SemanticResult] = []
    for post, sim in zip(posts, similarities):
        score = float(sim)
        results.append(
            SemanticResult(
                post_id=post.id,
                similarity=round(score, 4),
                action=_action_from_score(score),
            )
        )
    return results
