import json
import re
from typing import Any
from services import llm_cache
from openai import OpenAI
from services import llm_reasoner
from services.config import settings
from services import local_llm
from services.risk_classifier import classify_risk as _rule_based_classify

# ============================================================
# CONFIGURATION
# ============================================================

GEMINI_MODEL = getattr(
    settings,
    "LLM_MODEL_NAME",
    "gemini-3.8-flash"
)

# Qwen fallback usually runs on CPU. Keep each generation
# call small (few clauses at a time) so it stays fast and
# progress is visible, instead of one huge slow call.
QWEN_SUB_BATCH_SIZE = 3

GEMINI_CLIENT = None


# ============================================================
# GEMINI CLIENT
# PRIMARY CLASSIFIER
# ============================================================

if getattr(settings, "LLM_API_KEY", ""):

    try:

        GEMINI_CLIENT = OpenAI(
            base_url=settings.LLM_BASE_URL,
            api_key=settings.LLM_API_KEY,
            timeout=settings.LLM_TIMEOUT,
            max_retries=0,
        )
        print(
            f"[GEMINI] Primary model: {GEMINI_MODEL}"
        )

    except Exception as e:

        print(
            f"[GEMINI] Client initialization failed: {e}"
        )

        GEMINI_CLIENT = None

else:

    print(
        "[GEMINI] API key not configured."
    )


# ============================================================
# JSON EXTRACTION
# ============================================================

def _extract_json(
    text: str
) -> Any:

    if not text:

        raise ValueError(
            "Empty LLM response"
        )

    text = text.strip()

    # Remove markdown code fences
    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"\s*```$",
        "",
        text
    )

    # Direct JSON
    try:

        return json.loads(text)

    except Exception:

        pass

    # JSON object
    start = text.find("{")
    end = text.rfind("}")

    if (
        start >= 0
        and end > start
    ):

        candidate = text[
            start:end + 1
        ]

        try:

            return json.loads(
                candidate
            )

        except Exception:

            pass

    # JSON array
    start = text.find("[")
    end = text.rfind("]")

    if (
        start >= 0
        and end > start
    ):

        candidate = text[
            start:end + 1
        ]

        try:

            return json.loads(
                candidate
            )

        except Exception:

            pass

    raise ValueError(
        "Could not parse JSON from LLM response"
    )


# ============================================================
# UNKNOWN RESULT
#
# IMPORTANT:
# Never convert an LLM failure to LOW.
# ============================================================

def _unknown_result(
    reason: str
) -> dict:

    return {
        "risk_level": "UNKNOWN",
        "risk_score": None,
        "category": "Unclassified",
        "detected_issue": reason,
        "classifier": "llm_failed",
        "confidence": 0.0,
    }


# ============================================================
# NORMALIZE RESULT
# ============================================================

def _normalize_result(
    item: dict,
    default_category: str
) -> dict:

    if not isinstance(
        item,
        dict
    ):

        raise ValueError(
            "LLM result is not an object"
        )

    risk_level = str(
        item.get(
            "risk_level",
            "UNKNOWN"
        )
    ).upper().strip()

    if risk_level not in {
        "LOW",
        "MEDIUM",
        "HIGH"
    }:

        risk_level = "UNKNOWN"

    # --------------------------------------------------------
    # SCORE
    # --------------------------------------------------------

    try:

        score = int(
            float(
                item.get(
                    "risk_score"
                )
            )
        )

    except (
        TypeError,
        ValueError
    ):

        score = None

    if score is not None:

        score = max(
            0,
            min(
                score,
                100
            )
        )

    # --------------------------------------------------------
    # CATEGORY
    # --------------------------------------------------------

    category = str(
        item.get(
            "category",
            default_category
        )
        or default_category
    ).strip()

    # --------------------------------------------------------
    # ISSUE
    # --------------------------------------------------------

    issue = str(
        item.get(
            "detected_issue",
            ""
        )
        or ""
    ).strip()

    if not issue:

        if risk_level == "LOW":

            issue = (
                "No major legal risk identified."
            )

        elif risk_level == "MEDIUM":

            issue = (
                "Potential legal concern "
                "requiring review."
            )

        elif risk_level == "HIGH":

            issue = (
                "Significant legal risk identified."
            )

        else:

            issue = (
                "Unable to determine legal risk."
            )

    # --------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------

    try:

        confidence = float(
            item.get(
                "confidence",
                0.70
            )
        )

    except (
        TypeError,
        ValueError
    ):

        confidence = 0.70

    confidence = max(
        0.0,
        min(
            confidence,
            1.0
        )
    )

    return {
        "risk_level": risk_level,
        "risk_score": score,
        "category": category,
        "detected_issue": issue,
        "classifier": "llm_semantic",
        "confidence": confidence,
    }


# ============================================================
# COMMON CLASSIFICATION PROMPT
# ============================================================

def _build_prompt(
    clauses: list[str]
) -> str:

    blocks = []

    for index, clause in enumerate(
        clauses,
        start=1
    ):

        blocks.append(
            f"""
CLAUSE_ID: {index}

CLAUSE_TEXT:
{clause}
"""
        )

    joined = "\n".join(
        blocks
    )

    return f"""
You are the semantic legal-risk classification
engine for a legal document analysis system.

Classify EACH clause independently.

This is a semantic legal analysis task.

IMPORTANT RULES:

1. Understand the legal meaning and practical
   effect of the clause.

2. Do NOT depend only on keywords.

3. The document may contain completely new
   wording that you have never seen before.

4. Consider:
   - rights
   - obligations
   - financial exposure
   - termination
   - liability
   - indemnity
   - employment restrictions
   - privacy
   - authority
   - dispute resolution
   - unilateral discretion
   - penalties
   - payment obligations

5. Do NOT automatically classify a clause as HIGH
   just because it contains a concerning word.

6. HIGH means substantial legal, financial,
   employment, privacy, authority, liability,
   termination, or dispute-resolution exposure.

7. MEDIUM means a meaningful concern, ambiguity,
   imbalance, restriction, or review issue.

8. LOW means ordinary, balanced, administrative,
   definitional, procedural, or low-concern language.

9. Do NOT invent facts that are not present.

10. Risk level and confidence are different.
    Confidence represents how certain you are
    about your classification.

11. Return exactly ONE result for every input clause.

Return ONLY valid JSON.

Required format:

{{
  "results": [
    {{
      "clause_id": 1,
      "risk_level": "LOW|MEDIUM|HIGH",
      "risk_score": 0,
      "category": "short category",
      "detected_issue": "specific issue",
      "confidence": 0.0
    }}
  ]
}}

Risk score guidance:

LOW = 0-39
MEDIUM = 40-69
HIGH = 70-100

INPUT CLAUSES:

{joined}
"""


# ============================================================
# GEMINI CLASSIFICATION
# PRIMARY
# ============================================================

def _classify_with_gemini(
    clauses: list[str],
    default_category: str
) -> list[dict]:

    if not clauses:

        return []

    if GEMINI_CLIENT is None:

        raise RuntimeError(
            "Gemini client is not configured"
        )

    prompt = _build_prompt(
        clauses
    )

    response = (
        GEMINI_CLIENT
        .chat.completions.create(
            model=GEMINI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a semantic legal "
                        "risk classifier. "
                        "Return JSON only."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0.1,
            response_format={
                "type": "json_object"
            },
        )
    )

    raw = (
        response
        .choices[0]
        .message
        .content
    )

    data = _extract_json(
        raw
    )

    if isinstance(
        data,
        dict
    ):

        results = data.get(
            "results",
            []
        )

    elif isinstance(
        data,
        list
    ):

        results = data

    else:

        raise ValueError(
            "Unexpected Gemini JSON structure"
        )

    if not isinstance(
        results,
        list
    ):

        raise ValueError(
            "Gemini results is not a list"
        )

    if len(results) != len(
        clauses
    ):

        raise ValueError(
            f"Gemini returned "
            f"{len(results)} results for "
            f"{len(clauses)} clauses"
        )

    normalized = []

    for item in results:

        normalized.append(
            _normalize_result(
                item,
                default_category
            )
        )

    return normalized


# ============================================================
# QWEN GENERATION
# FALLBACK
# ============================================================

def _generate_with_qwen(
    prompt: str,
    max_new_tokens: int = 700
) -> str:

    return local_llm.generate(
        system_prompt=(
            "You are a semantic legal-risk "
            "classification engine. "
            "Return JSON only."
        ),
        user_prompt=prompt,
        max_new_tokens=max_new_tokens,
    )


# ============================================================
# QWEN CLASSIFICATION
# FALLBACK
# ============================================================

def _classify_with_qwen(
    clauses: list[str],
    default_category: str
) -> list[dict]:

    if not clauses:

        return []

    prompt = _build_prompt(
        clauses
    )

    raw = _generate_with_qwen(
        prompt,
        max_new_tokens=(
            250 * len(clauses)
        )
    )

    data = _extract_json(
        raw
    )

    if isinstance(
        data,
        dict
    ):

        results = data.get(
            "results",
            []
        )

    elif isinstance(
        data,
        list
    ):

        results = data

    else:

        raise ValueError(
            "Unexpected Qwen JSON structure"
        )

    if not isinstance(
        results,
        list
    ):

        raise ValueError(
            "Qwen results is not a list"
        )

    if len(results) != len(
        clauses
    ):

        raise ValueError(
            f"Qwen returned "
            f"{len(results)} results for "
            f"{len(clauses)} clauses"
        )

    normalized = []

    for item in results:

        normalized.append(
            _normalize_result(
                item,
                default_category
            )
        )

    # Mark actual fallback
    for item in normalized:

        item["classifier"] = (
            "local_qwen_fallback"
        )

    return normalized


# ============================================================
# CACHE
# ============================================================

_CACHE = {}


# ============================================================
# SINGLE CLAUSE
# ============================================================

def classify_semantic_risk(
    clause: str,
    default_category: str = "General Legal Clause"
) -> dict:

    clause = str(
        clause or ""
    ).strip()

    if not clause:

        return _unknown_result(
            "Empty clause"
        )

    if clause in _CACHE:

        return _CACHE[clause]

    # --------------------------------------------------------
    # 1. GEMINI PRIMARY
    # --------------------------------------------------------

    try:

        print(
            "[RISK] Trying Gemini primary..."
        )

        result = _classify_with_gemini(
            [clause],
            default_category
        )[0]

        _CACHE[clause] = result

        return result

    except Exception as e:

        print(
            "[GEMINI] Primary classification failed: "
            f"{e}"
        )

    # --------------------------------------------------------
    # 2. QWEN FALLBACK
    # --------------------------------------------------------

    try:

        print(
            "[RISK] Switching to local Qwen fallback..."
        )

        result = _classify_with_qwen(
            [clause],
            default_category
        )[0]

        _CACHE[clause] = result

        return result

    except Exception as e:

        print(
            "[QWEN] Fallback classification failed: "
            f"{e}"
        )

    # --------------------------------------------------------
    # 3. UNKNOWN
    #
    # NEVER LOW.
    # --------------------------------------------------------

    result = _unknown_result(
        "Both Gemini and local Qwen "
        "classification failed"
    )

    _CACHE[clause] = result

    return result

# ============================================================
# BATCH CLASSIFICATION
# ============================================================

def _cls_cache_key(clause: str) -> str:
    return llm_cache.make_key("cls", GEMINI_MODEL, clause)


def classify_semantic_risk_batch(
    clauses,
    default_category="General Legal Clause",
    batch_size=60
):
    if not clauses:
        return []

    clean_clauses = [str(c or "").strip() for c in clauses]

    if batch_size <= 0:
        batch_size = 60

    results = [None] * len(clean_clauses)
    pending = []

    for i, clause in enumerate(clean_clauses):
        cached = llm_cache.get(_cls_cache_key(clause)) if clause else None
        if cached:
            results[i] = cached
        else:
            pending.append(i)

    cached_count = len(clean_clauses) - len(pending)
    if cached_count:
        print(f"[CACHE] {cached_count}/{len(clean_clauses)} clauses served from cache (no Gemini call)")

    for start in range(0, len(pending), batch_size):
        idxs = pending[start:start + batch_size]
        batch = [clean_clauses[i] for i in idxs]

        print(f"[RISK] Semantic batch {start + 1}-{start + len(batch)} of {len(pending)} uncached")

        batch_results = None

        if llm_reasoner.is_gemini_available():
            try:
                print("[RISK] Using Gemini primary...")
                batch_results = llm_reasoner.call_with_transient_retry(
                    lambda: _classify_with_gemini(batch, default_category)
                )
                llm_cache.put_many([
                    (_cls_cache_key(clean_clauses[i]), r)
                    for i, r in zip(idxs, batch_results)
                    if r.get("risk_level") != "UNKNOWN"
                ])
            except Exception as e:
                print(f"[GEMINI] Batch failed: {e} — Gemini skipped for the rest of this document")
                llm_reasoner.mark_gemini_unavailable()
                batch_results = None

        if batch_results is None:
            print("[RISK] Using instant local rule-based fallback...")
            batch_results = []
            for clause in batch:
                rule_result = _rule_based_classify(clause)
                rule_result["classifier"] = "local_rule_fallback"
                if rule_result.get("category") == "General Legal Clause" and default_category:
                    rule_result["category"] = default_category
                batch_results.append(rule_result)

        for i, r in zip(idxs, batch_results):
            results[i] = r

    return results