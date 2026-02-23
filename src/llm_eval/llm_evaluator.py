import os
import ast
import json
import time
from typing import Any, Dict, List, Optional
from openai import OpenAI, APITimeoutError, APIConnectionError
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
import torch
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from src.llm_eval.utils import (
    VTGLabelSpace,
    calc_temporal_iou,
    extract_answer_content,
    extract_reasoning_text,
    jaccard_index_sets,
    strip_format_tags,
)


load_dotenv()

# Global SBERT model (lazy initialization)
_sbert_model = None

def get_sbert_model() -> SentenceTransformer:
    """
    Load Sentence BERT model (once per process).
    Uses CPU to prevent CUDA re-initialization errors in forked subprocesses.
    """
    global _sbert_model
    if _sbert_model is None:
        try:
            _sbert_model = SentenceTransformer('all-mpnet-base-v2', device='cpu')
        except Exception as e:
            raise RuntimeError(
                f"Failed to load SBERT model 'all-mpnet-base-v2'.\n"
                f"Check internet connection (for first-time download) "
                f"or ensure model files are accessible.\n"
                f"Original error: {e}"
            ) from e
    return _sbert_model

def get_model_from_env(default: str = "gpt-4o-mini") -> str:
    return os.getenv("OPENAI_MODEL", default)

def make_client_from_env() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing. Set it in .env or environment.")
    kwargs = {
        "api_key": api_key,
        "timeout": 120.0,
        "max_retries": 3,
    }
    base_url = os.getenv("OPENAI_BASE_URL")
    org = os.getenv("OPENAI_ORG_ID")
    project = os.getenv("OPENAI_PROJECT_ID")
    if base_url: kwargs["base_url"] = base_url
    if org: kwargs["organization"] = org
    if project: kwargs["project"] = project
    return OpenAI(**kwargs)


def _parse_py_dict_literal(text: str) -> Dict[str, Any]:
    """Safely parse a Python dictionary literal returned by a prompt."""
    return ast.literal_eval(text.strip())


def _call_openai_with_retry(client: OpenAI, model: str, messages: List[Dict[str, str]], max_retries: int = 5) -> str:
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(model=model, messages=messages)
            return resp.choices[0].message.content
        except (APITimeoutError, APIConnectionError) as e:
            if attempt == max_retries - 1:
                raise RuntimeError(f"OpenAI API call failed after {max_retries} attempts: {str(e)}") from None
            wait_time = 2 ** attempt
            print(f"API call failed (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s: {e}")
            time.sleep(wait_time)
        except Exception as e:
            raise RuntimeError(f"OpenAI API call failed: {type(e).__name__}: {str(e)}") from None

    raise RuntimeError(f"Failed to call OpenAI API after {max_retries} attempts")


# ---------------------------------------------------------
# Step 1: Relevance classification + top-1 span extraction
# ---------------------------------------------------------
def step1_relevance_and_top1_span(generated_response: str) -> Dict[str, Any]:
    system_prompt = """
You are an evaluator designed to assess the quality of a generated response to a video–text query pair in the context of Video Temporal Grounding.
Your evaluation has two components: (1) relevance classification and (2) Top-1 Temporal Span Extraction (only if relevant)

## INSTRUCTIONS:

### (1) Relevance Classification:
- If the Generated Response explicitly or implicitly states the query is unrelated to the video, classify 'irrelevant'.
- Otherwise classify 'relevant'.
- The decision should be made based on the semantic content and intent of the generated response, not merely on the presence of timestamps or time intervals.

### (2) Top-1 Temporal Span Extraction:
- Perform this step only when (1) has classified the response as 'relevant'.
- If the response was classified as 'irrelevant', simply return `span = None`.
- Identify the SINGLE most relevant time interval in the video that best answers the query, as indicated by the Generated Response.
- Prefer spans with strongest textual evidence (explicit timestamps > vague hints), highest specificity, and clear linkage to the query.
- If multiple spans are mentioned, select the best ONE; do not return multiple.
- Express the time interval in SECONDS as [start, end].
- If no usable time information is found, return span = None.

### OUTPUT:
Return ONLY a Python dictionary literal. No explanations.
Examples:
{'classification': 'relevant', 'span': [12.5, 18.0]}
{'classification': 'relevant', 'span': None}
{'classification': 'irrelevant', 'span': None}
""".strip()

    user_content = f"""
Generated Response:\n
{generated_response}
""".strip()

    model = get_model_from_env()
    client = make_client_from_env()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_content},
    ]
    text = _call_openai_with_retry(client, model, messages)
    return _parse_py_dict_literal(text)


# ---------------------------------------------------------
# Step 2: Category extraction from irrelevant response
# ---------------------------------------------------------
VTG_VALID_PATHS = {
    "Action/ActionSequence",
    "Action/FineGrainedAction",
    "Object/ObjectExistence",
    "Object/ObjectPartRelation",
    "Object/ObjectSpatialRelation",
    "Object/ObjectMoving",
    "Scene/SceneTransition",
    "Scene/SceneExistence",
    "Attribute/AttributeValue",
    "Attribute/Counting",
    "Attribute/Comparison",
}

SYSTEM_PROMPT = """You are a strict multi-label classifier.
Identify which reasoning categories from the video–text mismatch categories below are used or implied in a Generated Response that explains why a query is irrelevant to a video.
Select all applicable categories according to the meaning expressed in the response.

## Video–Text Mismatch Categories
- Action/ActionSequence — Change the temporal order of actions or causal relation (e.g., "A man pours water after drinking" vs "A man pours water before drinking")
- Action/FineGrainedAction — Replace an action with another that looks similar at a single moment but shows a distinct temporal behavior over time (e.g., "A woman opens a door" vs "A woman closes a door", "A man throws a ball" vs "A man catches a ball")
- Object/ObjectExistence — Add, remove, or swap an object identity or category (e.g., "A boy plays with a guitar" vs "A boy plays with a violin")
- Object/ObjectPartRelation — Change part–whole or accessory relations (e.g., "A person wears a watch" vs "A person wears a bracelet", "A man replaces a car's wheel" vs "A man replaces a motorcycle's wheel")
- Object/ObjectSpatialRelation — Change relative spatial position or direction (e.g., "A cat sits under the table" vs "A cat sits on the table")
- Object/ObjectMoving — Change the motion direction or trajectory of an object (e.g., "A ball rolls to the left" vs "A ball rolls to the right")
- Scene/SceneTransition — Change scene order or transition direction (e.g., "A man walks from the kitchen to the living room" vs "A man walks from the living room to the kitchen")
- Scene/SceneExistence — Replace the type of environment or background scene (e.g., "A woman cooks in the kitchen" vs "A woman cooks in the bathroom")
- Attribute/AttributeValue — Change intrinsic properties such as color, size, material, shape, or emotion (e.g., "A red car stops at the light" vs "A blue car stops at the light")
- Attribute/Counting — Change the number of objects or actions (e.g., "Two people are running" vs "Three people are running")
- Attribute/Comparison — Reverse comparative relations (e.g., "The man runs faster than the boy" vs "The man runs slower than the boy")

## Rules
1. Include a category only if it is clearly supported or implied by the reasoning.
2. Multiple categories may apply, but avoid redundant or speculative labels.
3. Use only the exact category paths listed above.
4. Ignore style, tone, or fluency — focus purely on reasoning content.
5. If none apply, return an empty list.

## Output Format
Return only a JSON array of strings containing the selected categories.
Examples:
["Object/ObjectExistence", "Attribute/Counting"]
If none apply: []
"""


def _few_shot_messages(few_shots: Optional[List[Dict[str, Any]]]) -> List[Dict[str, str]]:
    msgs: List[Dict[str, str]] = []
    if not few_shots:
        return msgs

    for fs in few_shots:
        inp = (fs.get("input") or "").strip()
        out = fs.get("output", None)
        if not inp or out is None:
            continue

        if isinstance(out, str):
            out_str = out.strip()
        else:
            out_str = json.dumps(out, ensure_ascii=False)

        msgs.append({"role": "user", "content": f"Generated Response: {inp}"})
        msgs.append({"role": "assistant", "content": out_str})
    return msgs


def step2_extract_categories(generated_response: str) -> List[str]:
    """
    Extract VTG taxonomy categories from the generated response.
    Returns a list of category path strings (e.g. ["Object/ObjectExistence"]).
    """
    if not generated_response or not generated_response.strip():
        return []

    model = get_model_from_env()
    client = make_client_from_env()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    few_shots = [
        {
            "input": "This query is somewhat inconsistent with the video segment and does not match the observed sequence of actions. The video shows her getting off the couch and then dancing, while the query places dancing before getting off, so no portion of the footage matches that order.",
            "output": [
                "Action/ActionSequence"
            ]
        },
        {
            "input": "This query does not align well with the video segment. The query states the topic is the East region, while the video segment features Peter introducing the South region, so the segment does not describe the East. The query characterizes the region as \"tightly contested,\" but in the part where Peter speaks the region is described as \"wide open,\" so the segment does not support the tighter-contested description; no part of the segment satisfies that wording.",
            "output": [
                "Object/ObjectExistence",
                "Attribute/AttributeValue"
            ]
        },
        {
            "input": "This query is clearly inconsistent with the video.</irrelevant_answer><object_objectexistence>The footage shows an American flag on the church exterior, while the query claims there is no flag, so that condition is not met.</object_objectexistence><scene_sceneexistence>The video shows the exterior of Mount of Olives Lutheran Church, but the query refers to the interior of a community center, so the scene does not match.</scene_sceneexistence><attribute_counting>The segment displays three crosses on the church exterior, yet the query mentions two crosses, so the numeric detail is incorrect and no part of the video satisfies it.",
            "output": [
                "Object/ObjectExistence",
                "Scene/SceneExistence",
                "Attribute/Counting"
            ]
        },
        {
            "input": "This query does not align well with the video segment. The footage displays armrests on the chair rather than wheels, so the listed part is incorrect. The segment clearly shows those parts present, so claiming their absence contradicts the material and no interval supports that claim.",
            "output": [
                "Object/ObjectPartRelation",
                "Negation/Negation"
            ]
        }
    ]
    messages += _few_shot_messages(few_shots)
    messages.append({"role": "user", "content": f"Generated Response: {generated_response.strip()}"})

    text = _call_openai_with_retry(client, model, messages).strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [p for p in parsed if p in VTG_VALID_PATHS]
        return []
    except Exception:
        return []


# ---------------------------------------------------------
# Step 3: Reasoning consistency scoring
# ---------------------------------------------------------
def step3_score_reasoning(
    generated_response: str,
    gt_response: str,
) -> Dict[str, Any]:
    system_prompt = """
You are an evaluator designed to assess the reasoning consistency between a Generated Response and a Ground Truth (GT) Response for a video–text query pair in the context of Video Temporal Grounding.

## TASK CONTEXT:
In this task, a query describes an event or action, and the model must identify the corresponding temporal segment in a video.
When the **Ground Truth Relevance** is `'irrelevant'`, it means that **the query is not related to the video content**.

### Important Note on Query Modification:
- The irrelevant query may have been created by modifying an original relevant query (e.g., changing object names, attributes, actions, etc.).
- The Generated Response may include a statement like "The original query before modification might be: [original query]" at the end.
- This statement provides context about what the original query was before modification, helping to understand the nature of the mismatch.
- When evaluating, focus on the reasoning about WHY the query is irrelevant, not on the presence or accuracy of the original query prediction.
- The original query prediction is supplementary information and should NOT significantly affect the score unless it directly contradicts the reasoning.

Your job is to evaluate how faithfully the Generated Response reproduces the reasoning and justification given in the GT Response for this irrelevance case.

---

## INSTRUCTIONS:

### Reasoning Consistency Evaluation:
- Evaluate how faithfully the Generated Response reproduces the reasoning and justification in the GT Response.
- The task applies only when the Ground Truth Relevance is `'irrelevant'`.
- A consistent response must preserve the GT's reasoning points — including all mismatch types, evidence, and contextual explanations — without contradicting or distorting their meaning.
- Omissions or contradictions of GT reasoning elements must be penalized.
- Additional reasoning or explanations should *not* be penalized as long as they are logically consistent and do not contradict the GT Response.
- If the Generated Response includes "The original query before modification might be: [...]", treat this as supplementary context, not as core reasoning to be evaluated.
- The reasoning must remain factually and logically compatible with the GT Response.
- Do not reward fluency, tone, or paraphrasing quality — focus purely on semantic and factual consistency.

### Scoring Scale (0–5):
The score may include decimal values (one decimal place) to reflect finer distinctions in reasoning consistency.

- 5.0 — Perfect Consistency: Fully aligned with GT reasoning; may include logically consistent elaborations that do not conflict with the GT.
- 4.0–4.9 — High Consistency: Largely aligned; may include minor paraphrasing or small consistent additions.
- 3.0–3.9 — Moderate Consistency: Captures most GT reasoning but omits or weakens key parts, or adds minor but non-conflicting content.
- 2.0–2.9 — Low Consistency: Misses multiple GT reasoning elements, or introduces reasoning partially inconsistent with GT logic.
- 1.0–1.9 — Minimal Consistency: Mostly diverges; significant omissions or conflicting explanations.
- 0.0–0.9 — No Consistency: Contradicts or ignores GT reasoning entirely.

### Evaluation Mindset:
- You MUST prioritize factual and logical alignment over stylistic similarity.
- Do NOT penalize harmless elaborations.
- Do NOT penalize the presence or content of "The original query before modification might be: [...]" statements.
- You MUST penalize any omission or contradiction of GT reasoning.
- You MUST NOT assign a score above 4.9 unless reasoning is perfectly consistent.

## OUTPUT:
Return ONLY a Python dictionary literal. No explanations.

Examples:
{'score': 4.0}
{'score': 1.5}
{'score': 3.7}

"""

    user_content = f"""
Generated Response:
{generated_response}

Ground Truth (GT) Response:
{gt_response}
""".strip()

    model = get_model_from_env()
    client = make_client_from_env()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_content},
    ]
    text = _call_openai_with_retry(client, model, messages)
    return _parse_py_dict_literal(text)


# ---------------------------------------------------------
# Step 4: SBERT semantic similarity
# ---------------------------------------------------------
def step4_sbert_similarity(
    generated_response: str,
    gt_response: str,
) -> Dict[str, Any]:
    """
    Compute semantic similarity between generated response and GT response
    using Sentence BERT cosine similarity.

    Returns:
        {'similarity': float} — cosine similarity in [0.0, 1.0]
    """
    if not generated_response or not generated_response.strip():
        return {"similarity": 0.0}
    if not gt_response or not gt_response.strip():
        return {"similarity": 0.0}

    model = get_sbert_model()

    embeddings = model.encode(
        [generated_response.strip(), gt_response.strip()],
        convert_to_tensor=True,
        show_progress_bar=False
    )

    similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
    similarity = max(0.0, min(1.0, similarity))

    return {"similarity": float(similarity)}


# Label space (initialized once at module level)
label_space = VTGLabelSpace.from_paths(VTG_VALID_PATHS)

# ---------------------------------------------------------
# Orchestration: four stages
#   Returns: {
#     'classification_result': bool,
#     'score': float,
#     'sbert_similarity': float,
#     'iou': float,
#     'pred_categories': List[str],
#     'jaccard_categories': Optional[float],
#     'per_label_counts': Optional[dict],
#   }
# ---------------------------------------------------------
def evaluate_four_stage(
    generated_response: str,
    gt_response: str,
    gt_relevance: str,
    gt_span: Optional[List[float]],
    gt_categories: List[str],
    cached_classification_correct: Optional[bool] = None,
    cached_iou: Optional[float] = None,
    cached_pred_categories: Optional[List[str]] = None
) -> Dict[str, Any]:
    # ---- Step 1: Relevance classification + span extraction ----
    if cached_classification_correct is not None:
        # Use cached classification result
        classification_result = cached_classification_correct
        if classification_result:
            pred_relevance = gt_relevance
        else:
            pred_relevance = "irrelevant" if gt_relevance == "relevant" else "relevant"

        # Extract pred_span only when needed for IoU (no cached IoU)
        if cached_iou is None and gt_relevance == "relevant" and pred_relevance == "relevant":
            generated_step1 = extract_answer_content(generated_response)
            s1 = step1_relevance_and_top1_span(generated_step1)
            pred_span = s1.get("span", None)
        else:
            pred_span = None
    else:
        generated_step1 = extract_answer_content(generated_response)
        s1 = step1_relevance_and_top1_span(generated_step1)
        pred_relevance = s1.get("classification")
        pred_span = s1.get("span", None)

        if pred_relevance not in ("relevant", "irrelevant"):
            print(f"Invalid classification: {pred_relevance}")
            pred_relevance = "irrelevant"

        classification_result = (pred_relevance == gt_relevance)

    # Default result
    result: Dict[str, Any] = {
        "classification_result": classification_result,
        "score": 0.0,
        "iou": 0.0,
        "pred_categories": [],
        "jaccard_categories": None,
        "per_label_counts": None,
    }

    # ---- IoU calculation ----
    if cached_iou is not None:
        result["iou"] = cached_iou
    else:
        if not classification_result:
            # Wrong relevance prediction → IoU = 0.0
            result["iou"] = 0.0
        else:
            if gt_relevance == "irrelevant":  # Both irrelevant
                result["iou"] = 1.0
            else:  # Both relevant → compute temporal IoU
                if (
                    isinstance(gt_span, list) and len(gt_span) == 2 and
                    isinstance(pred_span, list) and len(pred_span) == 2
                ):
                    result["iou"] = float(calc_temporal_iou(gt_span, pred_span))
                else:
                    result["iou"] = 0.0

    # ---- Steps 2–4 run only when GT is 'irrelevant' ----
    if gt_relevance == "irrelevant":
        if not gt_response:
            raise ValueError("gt_response is required when gt_relevance == 'irrelevant'.")

        gt_categories = gt_categories or []

        # ---- Step 2: Category extraction ----
        if cached_pred_categories is not None:
            pred_categories = cached_pred_categories
        else:
            if pred_relevance == "irrelevant":
                generated_step234 = extract_reasoning_text(generated_response)
                pred_categories = step2_extract_categories(generated_step234)
            else:
                # pred_relevance == 'relevant' → treat pred_categories as empty
                pred_categories = []

        result["pred_categories"] = pred_categories

        # Multi-label evaluation (sample Jaccard + per-label tp/fp/fn)
        jaccard = jaccard_index_sets(pred_categories, gt_categories)
        result["jaccard_categories"] = float(jaccard)

        y_true = label_space.encode(gt_categories)
        y_pred = label_space.encode(pred_categories)
        per_label_counts = label_space.new_counts()
        label_space.update_counts(per_label_counts, y_true, y_pred)
        result["per_label_counts"] = per_label_counts

        # ---- Step 3: Reasoning consistency score ----
        generated_step234 = extract_reasoning_text(generated_response)
        gt_processed = strip_format_tags(gt_response)
        s3 = step3_score_reasoning(generated_step234, gt_processed)
        result["score"] = float(s3.get("score", 0.0))

        # ---- Step 4: SBERT semantic similarity ----
        s4 = step4_sbert_similarity(generated_step234, gt_processed)
        result["sbert_similarity"] = s4.get("similarity", 0.0)

    return result


# Alias for backward compatibility
evaluate_three_stage = evaluate_four_stage
