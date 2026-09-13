# ShieldFeed Backend

FastAPI backend implementing the two ShieldFeed AI features:

- `POST /semantic-filter` — Semantic Trigger Filter
- `POST /toxicity` — Pre-Post Toxicity Check
- `GET /health` — deployment health check

## 1. Local setup

```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

## 2. Run locally

```bash
uvicorn app.main:app --reload --port 8000
```

Open http://localhost:8000/docs — FastAPI auto-generates an interactive
API tester (Swagger UI). Use it to try `/semantic-filter` and
`/toxicity` by hand before wiring up the frontend.

The first request to each AI endpoint will be slow (the model downloads
from Hugging Face and loads into memory). Every request after that is
fast, because both models are cached in memory for the life of the
process (see the `@lru_cache` in `semantic_service.py` /
`toxicity_service.py`).

## 3. Run tests

```bash
pytest
```

The semantic/toxicity tests call the real models (no mocking), so the
first run will be slow while models download. This is intentional for
a hackathon: we want to know the real models actually work, not just
that our code compiles.

## 4. Environment variables

See `.env.example` for the full list. The only one you *must* update
before final deployment is `ALLOWED_ORIGINS` — set it to your real
Vercel URL once the frontend is deployed, e.g.:

```
ALLOWED_ORIGINS=http://localhost:3000,https://shieldfeed.vercel.app
```

## 5. Deployment (Render — recommended for this stack)

1. Push this repo to GitHub.
2. On [render.com](https://render.com), create a **New Web Service**,
   connect the repo, and set:
   - **Root directory:** `backend`
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - **Environment variables:** copy every key from `.env.example` into
     Render's dashboard (Environment tab), with real values.
3. Deploy. First boot will be slow (models downloading) — this is
   normal. Watch the logs until you see Uvicorn's "Application startup
   complete."
4. Test the live URL immediately:
   ```bash
   curl https://<your-render-url>/health
   ```

### If you hit an out-of-memory error on the free tier

`torch` + two transformer models can be tight on 512MB instances. If
you see the process getting killed:

- Make sure `torch` is CPU-only (the pinned version in
  `requirements.txt` is fine on Render's Linux CPU instances by
  default — Render doesn't need the CUDA build).
- As a fallback, you can swap `toxicity_service.py` to call the
  Hugging Face **hosted** Inference API instead of loading the model
  locally (still an open-source pretrained model — just hosted by HF
  instead of by you). Ask the AI developer before making this switch,
  since it changes the service's internal behavior (network call
  instead of local inference) even though the API contract for
  `/toxicity` stays identical.

## 6. What NOT to touch without discussion

The three endpoints and their request/response shapes
(`docs/api.md`) are the contract the frontend is built against.
Changing field names or status codes breaks the frontend without
warning — coordinate first.
