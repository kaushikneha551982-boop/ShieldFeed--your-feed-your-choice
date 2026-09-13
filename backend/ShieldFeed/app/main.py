from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import base64
import io

from PIL import Image
import torch

from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer, util


# ==========================================
# CREATE FASTAPI APP
# ==========================================

app = FastAPI(title="Shield Field AI")


# ==========================================
# CORS
# ==========================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# ==========================================
# LOAD AI MODELS
# ==========================================

print("Loading AI models...")

text_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

clip_model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32"
)

clip_processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-base-patch32"
)


device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

clip_model.to(device)
clip_model.eval()

print("Models loaded on:", device)


# ==========================================
# REQUEST FORMAT
# ==========================================

class AnalyzeRequest(BaseModel):

    post_id: str

    preference: str

    text: str

    frames: List[str] = []


# ==========================================
# DECODE IMAGE
# ==========================================

def decode_image(data):

    try:

        if "," in data:

            data = data.split(",", 1)[1]


        raw = base64.b64decode(data)

        image = Image.open(
            io.BytesIO(raw)
        ).convert("RGB")

        return image


    except Exception as e:

        print(
            "Image decode error:",
            e
        )

        return None


# ==========================================
# TEXT SIMILARITY
# ==========================================

def calculate_text_similarity(
    text,
    preference
):

    if not text:

        return 0.0


    text_embedding = text_model.encode(
        text,
        convert_to_tensor=True
    )


    preference_embedding = text_model.encode(
        preference,
        convert_to_tensor=True
    )


    similarity = util.cos_sim(
        text_embedding,
        preference_embedding
    ).item()


    return float(
        max(0, similarity)
    )


# ==========================================
# ANALYZE VIDEO FRAMES
# ==========================================

def analyze_frames(
    frames,
    preference
):

    if not frames:

        return {
            "score": 0.0,
            "label": "no visual data"
        }


    images = []


    for frame in frames:

        image = decode_image(frame)

        if image is not None:

            images.append(image)


    if not images:

        return {
            "score": 0.0,
            "label": "no visual data"
        }


    labels = [

        "a dance video",

        "a person dancing",

        "people dancing",

        "a sports video",

        "a cooking video",

        "a nature video",

        "a technology video",

        "an educational video",

        "a normal social media video"

    ]


    inputs = clip_processor(
        text=labels,
        images=images,
        return_tensors="pt",
        padding=True
    )


    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }


    with torch.no_grad():

        outputs = clip_model(
            **inputs
        )


    image_text_scores = (
        outputs.logits_per_image
    )


    probabilities = (
        image_text_scores.softmax(dim=1)
    )


    # First three labels are dance-related
    dance_score = (
        probabilities[:, 0:3]
        .sum(dim=1)
        .mean()
        .item()
    )


    # ==========================================
    # CHECK USER PREFERENCE
    # ==========================================

    preference_inputs = clip_processor(
        text=[
            preference,
            "something unrelated to " + preference
        ],
        images=images,
        return_tensors="pt",
        padding=True
    )


    preference_inputs = {
        key: value.to(device)
        for key, value in preference_inputs.items()
    }


    with torch.no_grad():

        preference_outputs = clip_model(
            **preference_inputs
        )


    preference_probabilities = (
        preference_outputs
        .logits_per_image
        .softmax(dim=1)
    )


    preference_score = (
        preference_probabilities[:, 0]
        .mean()
        .item()
    )


    preference_lower = preference.lower()


    # ==========================================
    # DANCE DETECTION
    # ==========================================

    if (
        "dance" in preference_lower
        or "dancing" in preference_lower
    ):

        final_score = dance_score

        label = "dance detected"


    else:

        final_score = preference_score

        label = preference


    return {
        "score": float(final_score),
        "label": label
    }


# ==========================================
# ANALYZE ENDPOINT
# ==========================================

@app.post("/analyze")
async def analyze(
    request: AnalyzeRequest
):

    print("\n-----------------------------")

    print(
        "Post:",
        request.post_id
    )

    print(
        "Preference:",
        request.preference
    )

    print(
        "Text:",
        request.text[:100]
    )

    print(
        "Frames:",
        len(request.frames)
    )


    # ==========================================
    # TEXT ANALYSIS
    # ==========================================

    text_similarity = (
        calculate_text_similarity(
            request.text,
            request.preference
        )
    )


    # ==========================================
    # VISUAL ANALYSIS
    # ==========================================

    visual = analyze_frames(
        request.frames,
        request.preference
    )


    visual_score = visual["score"]


    preference_lower = request.preference.lower()


    # ==========================================
    # DANCE FILTER
    # ==========================================

    if (
        "dance" in preference_lower
        or "dancing" in preference_lower
    ):

        if visual_score >= 0.35:

            return {

                "post_id":
                    request.post_id,

                "action":
                    "FILTER",

                "reason":
                    "Dance video detected",

                "similarity":
                    visual_score,

                "confidence":
                    visual_score,

                "text_similarity":
                    text_similarity,

                "visual_similarity":
                    visual_score
            }


    # ==========================================
    # TEXT FILTER
    # ==========================================

    if text_similarity >= 0.55:

        return {

            "post_id":
                request.post_id,

            "action":
                "FILTER",

            "reason":
                "Content matches your shield",

            "similarity":
                text_similarity,

            "confidence":
                text_similarity,

            "text_similarity":
                text_similarity,

            "visual_similarity":
                visual_score
        }


    # ==========================================
    # VISUAL FILTER
    # ==========================================

    if (
        len(request.frames) > 0
        and visual_score >= 0.65
    ):

        return {

            "post_id":
                request.post_id,

            "action":
                "FILTER",

            "reason":
                "Visual content matches: "
                + request.preference,

            "similarity":
                visual_score,

            "confidence":
                visual_score,

            "text_similarity":
                text_similarity,

            "visual_similarity":
                visual_score
        }


    # ==========================================
    # SHOW CONTENT
    # ==========================================

    return {

        "post_id":
            request.post_id,

        "action":
            "SHOW",

        "reason":
            "No shield triggered",

        "similarity":
            max(
                text_similarity,
                visual_score
            ),

        "confidence":
            max(
                text_similarity,
                visual_score
            ),

        "text_similarity":
            text_similarity,

        "visual_similarity":
            visual_score
    }


# ==========================================
# ROOT
# ==========================================

@app.get("/")
def root():

    return {

        "status":
            "Shield Field AI backend running",

        "model":
            "Text + CLIP vision"
    }