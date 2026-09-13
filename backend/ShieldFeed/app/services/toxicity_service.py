"""
Pre-Post Toxicity Check service.

Loads a pretrained, open-source toxicity classifier ONCE (cached after
first use) and exposes a single classify() function that returns a
toxicity probability plus a SAFE/WARN decision.

No commercial LLM is used for the actual moderation decision - the
classifier here is the whole story. An LLM could optionally be added
later purely to generate a "rephrase" suggestion once a comment is
flagged WARN, but that is a separate, optional feature.
"""

from functools import lru_cache
from typing import Tuple

from transformers import pipeline

from app.config import settings


@lru_cache(maxsize=1)
def get_classifier():
    return pipeline(
        "text-classification",
        model=settings.TOXICITY_MODEL_NAME,
        top_k=None,  # return scores for every label, not just the top one
    )


def classify(text: str) -> Tuple[float, str]:
    classifier = get_classifier()
    predictions = classifier(text)[0]

    # martin-ha/toxic-comment-model returns labels like
    # "toxic" and "non-toxic", each with its own confidence score.
    toxic_score = None
    for pred in predictions:
        label = pred["label"].lower()
        if "toxic" in label and "non" not in label:
            toxic_score = pred["score"]
            break

    if toxic_score is None:
        # Fallback for models with generic labels (LABEL_0/LABEL_1 etc.):
        # take the highest-confidence prediction as a best-effort score.
        toxic_score = max(p["score"] for p in predictions)

    action = "WARN" if toxic_score >= settings.TOXICITY_WARN_THRESHOLD else "SAFE"
    return round(toxic_score, 4), action
