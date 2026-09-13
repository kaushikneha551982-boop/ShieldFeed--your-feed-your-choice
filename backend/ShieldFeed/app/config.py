"""
Central configuration for ShieldFeed backend.

Everything here can be overridden with environment variables
(see backend/.env.example) so we never hardcode secrets or
environment-specific values like the deployed frontend URL.
"""

import os


class Settings:
    PROJECT_NAME: str = "ShieldFeed"

    # Comma-separated list of allowed frontend origins.
    # Locally this defaults to the Next.js dev server.
    # In production, set this to your actual Vercel URL, e.g.:
    #   ALLOWED_ORIGINS=https://shieldfeed.vercel.app
    ALLOWED_ORIGINS: list[str] = os.getenv(
        "ALLOWED_ORIGINS", "http://localhost:3000"
    ).split(",")

    # --- Semantic Trigger Filter thresholds (cosine similarity, 0..1) ---
    # similarity >= FILTER_THRESHOLD          -> FILTER (hide, offer reveal)
    # WARN_THRESHOLD <= similarity < FILTER    -> WARN   (show with a warning)
    # similarity < WARN_THRESHOLD              -> SHOW   (show normally)
    SEMANTIC_FILTER_THRESHOLD: float = float(
        os.getenv("SEMANTIC_FILTER_THRESHOLD", 0.5)
    )
    SEMANTIC_WARN_THRESHOLD: float = float(
        os.getenv("SEMANTIC_WARN_THRESHOLD", 0.3)
    )

    # --- Pre-Post Toxicity Check threshold (probability, 0..1) ---
    TOXICITY_WARN_THRESHOLD: float = float(
        os.getenv("TOXICITY_WARN_THRESHOLD", 0.5)
    )

    # Pretrained, open-source models (no training required - loaded as-is)
    SEMANTIC_MODEL_NAME: str = os.getenv(
        "SEMANTIC_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
    )
    TOXICITY_MODEL_NAME: str = os.getenv(
        "TOXICITY_MODEL_NAME", "martin-ha/toxic-comment-model"
    )


settings = Settings()
