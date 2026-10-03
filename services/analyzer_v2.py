from services.document_loader import extract_text_from_pdf
from services.document_cleaner import clean_legal_text
from services.clause_extractor import extract_clauses
from services.semantic_risk_classifier import classify_semantic_risk_batch
from services.legal_risk_calibrator import calibrate_risk
from services.llm_reasoner import (
    generate_reasoning,
    generate_reasoning_batch,
    reset_gemini_availability,
)
from services.negotiation_engine import suggest_negotiation
from services.missing_clause_detector import detect_missing_clauses
from services.risk_score import verdict
from services.config import settings


# ============================================================
# CONFIGURATION
# ============================================================

SIMILARITY_THRESHOLD = 0.50
TOP_K = 3

# Gemini semantic classification happens in batches.
#
# 150 clauses / 30 = approximately 5 classification
# requests instead of 150 requests.
CLASSIFICATION_BATCH_SIZE = 60

# HIGH-risk reasoning is also batched (see analyze_document_v2,
# Section 8). This keeps Gemini reasoning calls low even on
# documents with many HIGH-risk clauses.
REASONING_BATCH_SIZE = 10


# ============================================================
# GENERAL HELPERS
# ============================================================

def safe_string(value, default=""):
    if value is None:
        return default
    return str(value).strip()


def normalize_risk_level(value):
    level = safe_string(value, "UNKNOWN").upper()

    if level not in {"LOW", "MEDIUM", "HIGH", "UNKNOWN"}:
        # An LLM failure or unparseable value must stay
        # visible as UNKNOWN, never silently become MEDIUM.
        return "UNKNOWN"

    return level


def safe_risk_score(value, default=50):
    try:
        score = int(float(value))
    except (TypeError, ValueError):
        score = default

    return max(0, min(score, 100))


# ============================================================
# TARGETED NORMALIZATION
#
# IMPORTANT:
# These checks DO NOT perform primary classification.
#
# Gemini already performs semantic classification.
#
# These checks only normalize obvious outputs and keep
# important legal categories consistent.
# ============================================================

def normalize_risk_result(clause_text: str, risk_result: dict) -> dict:
    if not isinstance(risk_result, dict):
        risk_result = {}

    result = dict(risk_result)

    text = safe_string(clause_text)
    lower_text = text.lower()

    result["risk_level"] = normalize_risk_level(result.get("risk_level", "UNKNOWN"))

    if result["risk_level"] == "UNKNOWN":
        # Keep score unset instead of inventing a number.
        result["risk_score"] = result.get("risk_score")
    else:
        result["risk_score"] = safe_risk_score(result.get("risk_score", 50))

    result["category"] = safe_string(
        result.get("category", "General Legal Clause"),
        "General Legal Clause",
    )

    result["detected_issue"] = safe_string(
        result.get("detected_issue", "Potential legal risk"),
        "Potential legal risk",
    )

    result["classifier"] = safe_string(
        result.get("classifier", "unknown"),
        "unknown",
    )

    # --------------------------------------------------------
    # Termination
    # --------------------------------------------------------

    has_termination = any(
        word in lower_text
        for word in ("terminate", "termination", "terminating")
    )

    no_notice = (
        "without notice" in lower_text
        or "without providing notice" in lower_text
        or "without any notice" in lower_text
        or "no notice" in lower_text
    )

    immediate_termination = (
        "terminate immediately" in lower_text
        or "immediate termination" in lower_text
        or "immediately terminate" in lower_text
    )

    sole_discretion = (
        "sole discretion" in lower_text
        or "solely at the discretion" in lower_text
        or "at its discretion" in lower_text
    )

    no_cure = (
        "without opportunity to cure" in lower_text
        or "without an opportunity to cure" in lower_text
        or "no opportunity to cure" in lower_text
        or "without cure period" in lower_text
    )

    if has_termination:
        result["category"] = "Termination"

        if immediate_termination and sole_discretion and no_notice:
            result["detected_issue"] = "Sole discretion termination"
        elif immediate_termination and no_notice:
            result["detected_issue"] = "Immediate termination without notice"
        elif sole_discretion:
            result["detected_issue"] = "Sole discretion termination"
        elif no_notice:
            result["detected_issue"] = "Termination without notice"
        elif no_cure:
            result["detected_issue"] = "Termination without cure opportunity"

    # --------------------------------------------------------
    # Unilateral amendment
    # --------------------------------------------------------

    unilateral_change = (
        "unilaterally amend" in lower_text
        or "unilateral amendment" in lower_text
        or "unilaterally modify" in lower_text
        or "unilateral modification" in lower_text
    )

    if unilateral_change:
        result["category"] = "Amendment"
        result["detected_issue"] = "Unilateral modification"

    # --------------------------------------------------------
    # Unlimited liability
    # --------------------------------------------------------

    if "unlimited liability" in lower_text or "unlimited liabilities" in lower_text:
        result["category"] = "Liability"
        result["detected_issue"] = "Unlimited liability"

    # --------------------------------------------------------
    # Uncapped indemnity
    # --------------------------------------------------------

    if "uncapped indemnity" in lower_text or "unlimited indemnity" in lower_text:
        result["category"] = "Indemnity"
        result["detected_issue"] = "Uncapped indemnity"

    # --------------------------------------------------------
    # Salary/payment withholding
    # --------------------------------------------------------

    if (
        "withhold salary" in lower_text
        or "withhold payment" in lower_text
        or "salary may be withheld" in lower_text
        or "payment may be withheld" in lower_text
    ):
        result["category"] = "Payment"
        result["detected_issue"] = "Salary/payment withholding"

    return result


# ============================================================
# OVERALL DOCUMENT SCORE
# ============================================================

def calculate_overall_score(results: list, missing_clauses: list) -> int:
    if not results:
        return 0

    total_score = 0
    high_count = 0
    scored_clauses = 0

    for result in results:
        level = normalize_risk_level(result.get("risk_level", "UNKNOWN"))

        # UNKNOWN clauses are excluded from the numeric
        # average instead of being silently treated as LOW.
        # They stay visible in the per-clause report.
        if level == "UNKNOWN":
            continue

        score = safe_risk_score(result.get("risk_score", 20), default=20)

        total_score += score
        scored_clauses += 1

        if level == "HIGH":
            high_count += 1

    total_clauses = len(results)

    average_score = (total_score / scored_clauses) if scored_clauses else 0
    high_ratio = high_count / total_clauses

    high_risk_boost = 0

    if high_count >= 10:
        high_risk_boost += 10
    elif high_count >= 5:
        high_risk_boost += 8
    elif high_count >= 3:
        high_risk_boost += 5

    if high_ratio >= 0.30:
        high_risk_boost += 8
    elif high_ratio >= 0.15:
        high_risk_boost += 5

    critical_missing_keywords = [
        "termination",
        "dispute resolution",
        "governing law",
        "jurisdiction",
        "limitation of liability",
        "confidentiality",
        "data protection",
        "privacy",
    ]

    missing_penalty = 0

    if not isinstance(missing_clauses, list):
        missing_clauses = []

    for missing in missing_clauses:
        if not isinstance(missing, dict):
            continue

        name = (
            missing.get("missing_clause")
            or missing.get("name")
            or missing.get("clause_name")
            or missing.get("title")
            or "Missing Clause"
        )

        name_lower = safe_string(name).lower()

        if any(keyword in name_lower for keyword in critical_missing_keywords):
            missing_penalty += 4
        else:
            missing_penalty += 2

    missing_penalty = min(missing_penalty, 15)

    final_score = round(average_score + high_risk_boost + missing_penalty)

    return min(final_score, 100)


# ============================================================
# RAG
# ============================================================

def get_similar_clauses_for_risk(clause_text: str, category: str = None) -> list:
    if not getattr(settings, "ENABLE_RAG", False):
        return []

    clause_text = safe_string(clause_text)

    if not clause_text:
        return []

    backend = safe_string(getattr(settings, "RAG_BACKEND", "faiss"), "faiss").lower()

    # --------------------------------------------------------
    # BM25
    # --------------------------------------------------------

    if backend == "bm25":
        try:
            from services.bm25_rag import search_similar_clauses_bm25

            retrieved = search_similar_clauses_bm25(
                clause_text, top_k=TOP_K, category=category
            )

            if not isinstance(retrieved, list):
                return []

            return retrieved[:TOP_K]

        except Exception as e:
            print(f"[RAG] BM25 retrieval failed: {e}")
            return []

    # --------------------------------------------------------
    # FAISS
    # --------------------------------------------------------

    try:
        from services.embedder import generate_embedding
        from services.vector_store import vector_store

        embedding = generate_embedding(clause_text)
        retrieved = vector_store.search(embedding, top_k=TOP_K * 4)

    except Exception as e:
        print(f"[RAG] FAISS retrieval failed: {e}")
        return []

    if not isinstance(retrieved, list):
        return []

    filtered = []
    seen = set()

    for item in retrieved:
        if not isinstance(item, dict):
            continue

        try:
            score = float(item.get("similarity_score", 0))
        except (TypeError, ValueError):
            score = 0

        if score < SIMILARITY_THRESHOLD:
            continue

        retrieved_clause = safe_string(item.get("clause_text", ""))

        if not retrieved_clause:
            continue

        key = " ".join(retrieved_clause.lower().split())

        if key in seen:
            continue

        seen.add(key)

        item_category = safe_string(item.get("clause_type", ""))
        item_risk = safe_string(item.get("risk_label", "UNKNOWN"), "UNKNOWN")

        category_match = True

        if category and safe_string(category).lower() not in {
            "general clause",
            "general legal clause",
            "unknown",
            "",
        }:
            category_match = safe_string(category).lower() in item_category.lower()

        item["retrieval_backend"] = "faiss"
        item["category_match"] = category_match
        item["risk_label_available"] = item_risk.upper() != "UNKNOWN"

        filtered.append(item)

    filtered.sort(
        key=lambda x: (
            x.get("category_match", False),
            x.get("risk_label_available", False),
            x.get("similarity_score", 0),
        ),
        reverse=True,
    )

    return filtered[:TOP_K]


# ============================================================
# LOCAL REASONING
# ============================================================

def low_risk_reasoning(clause_text: str) -> dict:
    return {
        "reason": (
            "This clause does not appear to create "
            "a significant legal risk based on the "
            "semantic assessment."
        ),
        "action": (
            "Review the commercial terms and surrounding "
            "clauses before signing."
        ),
        "safer_clause": clause_text,
    }


def medium_risk_reasoning(clause_text: str, risk_result: dict) -> dict:
    detected_issue = safe_string(
        risk_result.get("detected_issue", "a potential contractual concern"),
        "a potential contractual concern",
    )

    return {
        "reason": (
            "This clause requires additional review "
            f"because it may involve {detected_issue.lower()}."
        ),
        "action": (
            "Clarify the scope, conditions, timelines, "
            "and approval process before signing."
        ),
        "safer_clause": clause_text,
    }


def high_risk_fallback_reasoning(clause_text: str, risk_result: dict) -> dict:
    detected_issue = safe_string(
        risk_result.get("detected_issue", "significant contractual exposure"),
        "significant contractual exposure",
    )

    return {
        "reason": (
            "This clause was semantically classified "
            "as HIGH risk because it may create "
            f"significant exposure related to {detected_issue.lower()}."
        ),
        "action": (
            "Review and negotiate this clause with "
            "appropriate legal guidance before signing."
        ),
        "safer_clause": clause_text,
    }


# ============================================================
# HIGH RISK — RAG + GEMINI (single clause path)
# ============================================================

def generate_high_risk_reasoning(clause_text: str, risk_result: dict) -> tuple:
    try:
        similar_clauses = get_similar_clauses_for_risk(
            clause_text, category=risk_result.get("category")
        )
    except Exception as rag_error:
        print(f"[RAG] Retrieval failed: {rag_error}")
        similar_clauses = []

    try:
        reasoning = generate_reasoning(
            clause=clause_text,
            risk_result=risk_result,
            similar_clauses=similar_clauses,
        )

        if not isinstance(reasoning, dict):
            raise ValueError("LLM reasoning returned invalid response type")

        return reasoning, similar_clauses

    except Exception as llm_error:
        print(f"[LLM] HIGH reasoning failed: {llm_error}")
        return high_risk_fallback_reasoning(clause_text, risk_result), similar_clauses


# ============================================================
# SINGLE CLAUSE ANALYSIS
# ============================================================

def analyze_clause_v2(clause: str) -> dict:
    clause = safe_string(clause)

    if not clause:
        return {
            "original_clause": "",
            "risk_level": "LOW",
            "risk_score": 0,
            "category": "General Legal Clause",
            "detected_issue": "Empty clause",
            "classifier": "validation",
            "confidence": 1.0,
            "reason": "No clause text was provided.",
            "action": "Provide a valid legal clause.",
            "safer_clause": "",
            "negotiation_advice": {},
            "similar_clauses": [],
        }

    # --------------------------------------------------------
    # SEMANTIC CLASSIFICATION
    # --------------------------------------------------------

    try:
        risk_result = classify_semantic_risk_batch(
            [clause],
            default_category="General Legal Clause",
            batch_size=1,
        )[0]

    except Exception as e:
        print(f"[RISK] Semantic classification failed: {e}")

        # No rule-based fallback. Risk classification is
        # LLM-only (Gemini -> Qwen). If both failed inside
        # classify_semantic_risk_batch this is unreachable;
        # this branch only covers an unexpected local error.
        risk_result = {
            "risk_level": "UNKNOWN",
            "risk_score": None,
            "category": "Unclassified",
            "detected_issue": "Risk classification failed unexpectedly.",
            "classifier": "llm_failed",
            "confidence": 0.0,
        }

    # --------------------------------------------------------
    # CALIBRATION
    # --------------------------------------------------------

    try:
        risk_result = calibrate_risk(clause, risk_result)
    except Exception as e:
        print(f"[RISK] Calibration failed: {e}")

    risk_result = normalize_risk_result(clause, risk_result)

    # --------------------------------------------------------
    # NEGOTIATION
    # --------------------------------------------------------

    negotiation = suggest_negotiation(risk_result)

    # --------------------------------------------------------
    # REASONING
    # --------------------------------------------------------

    risk_level = normalize_risk_level(risk_result.get("risk_level"))

    if risk_level == "LOW":
        reasoning = low_risk_reasoning(clause)
        similar_clauses = []

    elif risk_level == "MEDIUM":
        reasoning = medium_risk_reasoning(clause, risk_result)
        similar_clauses = []

    else:
        reasoning, similar_clauses = generate_high_risk_reasoning(clause, risk_result)

    return {
        "original_clause": clause,
        "risk_level": risk_result["risk_level"],
        "risk_score": risk_result["risk_score"],
        "category": risk_result["category"],
        "detected_issue": risk_result["detected_issue"],
        "classifier": risk_result.get("classifier", "unknown"),
        "confidence": risk_result.get("confidence"),
        "reason": safe_string(
            reasoning.get("reason", "Risk requires further legal review."),
            "Risk requires further legal review.",
        ),
        "action": safe_string(
            reasoning.get("action", "Review this clause before signing."),
            "Review this clause before signing.",
        ),
        "safer_clause": safe_string(
            reasoning.get("safer_clause", clause),
            clause,
        ),
        "negotiation_advice": negotiation,
        "similar_clauses": similar_clauses,
    }


# ============================================================
# FULL DOCUMENT ANALYSIS
# ============================================================

def analyze_document_v2(pdf_path: str) -> dict:

    # FIX 1: reset_gemini_availability() must run once PER
    # REQUEST (per PDF), not once at module import time.
    # Calling it here — as the first line inside the function —
    # ensures every new document gets a fresh chance at Gemini,
    # instead of staying stuck on Qwen fallback until the whole
    # server process is restarted.
    reset_gemini_availability()

    # ========================================================
    # 1. EXTRACT PDF
    # ========================================================

    extracted = extract_text_from_pdf(pdf_path)

    if isinstance(extracted, dict):
        raw_text = extracted.get("raw_text", "")
        extraction_method = extracted.get("extraction_method", "unknown")
    else:
        raw_text = extracted
        extraction_method = "pymupdf_or_ocr"

    raw_text = safe_string(raw_text)

    # ========================================================
    # 2. CLEAN
    # ========================================================

    clean_text = clean_legal_text(raw_text)
    clean_text = safe_string(clean_text)

    # ========================================================
    # 3. CLAUSES
    # ========================================================

    all_clauses = extract_clauses(clean_text)

    if not isinstance(all_clauses, list):
        all_clauses = []

    total_clauses_extracted = len(all_clauses)

    # ========================================================
    # 4. ANALYSIS LIMIT
    # ========================================================

    max_clauses = getattr(settings, "MAX_ANALYSIS_CLAUSES", 0)

    try:
        max_clauses = int(max_clauses)
    except (TypeError, ValueError):
        max_clauses = 0

    if max_clauses > 0:
        clauses = all_clauses[:max_clauses]
    else:
        clauses = all_clauses

    analysis_limited = len(clauses) < total_clauses_extracted

    # ========================================================
    # 5. CLAUSE TEXTS
    # ========================================================

    clause_texts = []

    for clause_obj in clauses:
        if not isinstance(clause_obj, dict):
            clause_texts.append("")
            continue

        clause_texts.append(safe_string(clause_obj.get("text", "")))

    # ========================================================
    # 6. SEMANTIC GEMINI CLASSIFICATION
    #
    # IMPORTANT:
    # Gemini classification is BATCHED.
    #
    # 150 clauses / 30 ≈ 5 requests.
    # ========================================================

    try:
        risk_results = classify_semantic_risk_batch(
            clause_texts,
            default_category="General Legal Clause",
            batch_size=CLASSIFICATION_BATCH_SIZE,
        )

        if not isinstance(risk_results, list) or len(risk_results) != len(clause_texts):
            raise ValueError(
                "Semantic classifier returned an invalid number of results."
            )

    except Exception as e:
        print(f"[RISK] Semantic batch classification failed: {e}")

        # No rule-based fallback. Risk classification is
        # LLM-only (Gemini -> Qwen, handled inside
        # classify_semantic_risk_batch). If that call itself
        # raised, mark every clause UNKNOWN instead of
        # guessing MEDIUM.
        risk_results = [
            {
                "risk_level": "UNKNOWN",
                "risk_score": None,
                "category": "Unclassified",
                "detected_issue": "Risk classification failed unexpectedly.",
                "classifier": "llm_failed",
                "confidence": 0.0,
            }
            for _ in clause_texts
        ]

    # ========================================================
    # 7. CALIBRATE
    # ========================================================

    calibrated_results = []

    for index, result in enumerate(risk_results):
        clause_text = clause_texts[index]

        try:
            result = calibrate_risk(clause_text, result)
        except Exception as e:
            print(f"[RISK] Calibration failed for clause {index + 1}: {e}")

        result = normalize_risk_result(clause_text, result)
        calibrated_results.append(result)

    risk_results = calibrated_results

    # ========================================================
    # 8. CLAUSE REASONING
    #
    # FIX 3: LOW/MEDIUM clauses are reasoned locally (no LLM
    # call). HIGH-risk clauses are NOT reasoned one-by-one
    # anymore — they are collected into `high_items` and sent
    # to Gemini in batches of REASONING_BATCH_SIZE afterwards.
    # This is what cuts a 20-call reasoning pass down to ~4 calls.
    # ========================================================

    results = [None] * len(clauses)
    high_items = []    # collected HIGH-risk items for batch reasoning
    high_indices = []  # matching index into `results`

    for index, clause_obj in enumerate(clauses):

        if not isinstance(clause_obj, dict):
            continue

        clause_text = safe_string(clause_obj.get("text", ""))
        risk_result = normalize_risk_result(clause_text, risk_results[index])
        risk_level = normalize_risk_level(risk_result.get("risk_level", "MEDIUM"))
        negotiation = suggest_negotiation(risk_result)

        if risk_level == "LOW":
            reasoning = low_risk_reasoning(clause_text)
            similar_clauses = []

        elif risk_level == "MEDIUM":
            reasoning = medium_risk_reasoning(clause_text, risk_result)
            similar_clauses = []

        else:
            # Defer HIGH clauses for batch reasoning below.
            try:
                similar_clauses = get_similar_clauses_for_risk(
                    clause_text, category=risk_result.get("category")
                )
            except Exception as rag_error:
                print(f"[RAG] Retrieval failed: {rag_error}")
                similar_clauses = []

            high_items.append(
                {
                    "clause": clause_text,
                    "risk_result": risk_result,
                    "similar_clauses": similar_clauses,
                }
            )
            high_indices.append(index)

            results[index] = {
                "clause_id": clause_obj.get("clause_id", index + 1),
                "heading": clause_obj.get("heading", ""),
                "original_clause": clause_text,
                "risk_level": risk_result["risk_level"],
                "risk_score": risk_result["risk_score"],
                "category": risk_result["category"],
                "detected_issue": risk_result["detected_issue"],
                "classifier": risk_result.get("classifier", "unknown"),
                "confidence": risk_result.get("confidence"),
                "negotiation_advice": negotiation,
                "similar_clauses": similar_clauses,
                # reason/action/safer_clause filled in after batch reasoning below
            }
            continue

        results[index] = {
            "clause_id": clause_obj.get("clause_id", index + 1),
            "heading": clause_obj.get("heading", ""),
            "original_clause": clause_text,
            "risk_level": risk_result["risk_level"],
            "risk_score": risk_result["risk_score"],
            "category": risk_result["category"],
            "detected_issue": risk_result["detected_issue"],
            "classifier": risk_result.get("classifier", "unknown"),
            "confidence": risk_result.get("confidence"),
            "reason": safe_string(
                reasoning.get("reason", "This clause requires legal review."),
                "This clause requires legal review.",
            ),
            "action": safe_string(
                reasoning.get("action", "Review this clause before signing."),
                "Review this clause before signing.",
            ),
            "safer_clause": safe_string(
                reasoning.get("safer_clause", clause_text),
                clause_text,
            ),
            "negotiation_advice": negotiation,
            "similar_clauses": similar_clauses,
        }

    # --------------------------------------------------------
    # BATCH REASONING FOR ALL HIGH-RISK CLAUSES
    # --------------------------------------------------------

    high_reasoning_calls = 0

    if high_items:
        print(f"[HIGH] Batch reasoning for {len(high_items)} HIGH-risk clauses")

        for start in range(0, len(high_items), REASONING_BATCH_SIZE):
            batch = high_items[start:start + REASONING_BATCH_SIZE]
            batch_indices = high_indices[start:start + REASONING_BATCH_SIZE]

            high_reasoning_calls += 1

            try:
                batch_reasonings = generate_reasoning_batch(batch)
            except Exception as e:
                print(f"[HIGH] Batch reasoning failed entirely: {e} — using fallback text")
                batch_reasonings = [
                    high_risk_fallback_reasoning(item["clause"], item["risk_result"])
                    for item in batch
                ]

            for result_index, reasoning in zip(batch_indices, batch_reasonings):
                r = results[result_index]
                r["reason"] = safe_string(
                    reasoning.get("reason", "This clause requires legal review."),
                    "This clause requires legal review.",
                )
                r["action"] = safe_string(
                    reasoning.get("action", "Review this clause before signing."),
                    "Review this clause before signing.",
                )
                r["safer_clause"] = safe_string(
                    reasoning.get("safer_clause", r["original_clause"]),
                    r["original_clause"],
                )

    results = [r for r in results if r is not None]

    # ========================================================
    # 9. MISSING CLAUSES
    # ========================================================

    try:
        missing_clauses = detect_missing_clauses(clean_text)

        if not isinstance(missing_clauses, list):
            missing_clauses = []

    except Exception as e:
        print(f"[MISSING] Detection failed: {e}")
        missing_clauses = []

    # ========================================================
    # 10. OVERALL SCORE
    # ========================================================

    overall_score = calculate_overall_score(results, missing_clauses)

    # ========================================================
    # 11. VERDICT
    # ========================================================

    try:
        overall_verdict = verdict(overall_score)

    except Exception as e:
        print(f"[RISK] Verdict calculation failed: {e}")

        if overall_score >= 75:
            overall_verdict = "HIGH"
        elif overall_score >= 40:
            overall_verdict = "MEDIUM"
        else:
            overall_verdict = "LOW"

    # ========================================================
    # 12. COUNTS
    # ========================================================

    high_count = sum(
        1 for result in results
        if normalize_risk_level(result.get("risk_level")) == "HIGH"
    )

    medium_count = sum(
        1 for result in results
        if normalize_risk_level(result.get("risk_level")) == "MEDIUM"
    )

    low_count = sum(
        1 for result in results
        if normalize_risk_level(result.get("risk_level")) == "LOW"
    )

    # ========================================================
    # 13. FINAL REPORT
    # ========================================================

    return {
        "version": "3.0",
        "architecture": (
            "Gemini semantic risk classification in batches + "
            "legal risk calibration + "
            "deterministic safety normalization + "
            "FAISS/BM25 RAG + "
            "batched Gemini reasoning for HIGH-risk clauses"
        ),
        "classification_model": settings.LLM_MODEL_NAME,
        "extraction_method": extraction_method,
        "cleaned_text": clean_text,
        "total_clauses": len(clauses),
        "total_clauses_extracted": total_clauses_extracted,
        "analysis_limited": analysis_limited,
        "llm_reasoning_calls": high_reasoning_calls,
        "high_risk": high_count,
        "medium_risk": medium_count,
        "low_risk": low_count,
        "overall_risk_score": overall_score,
        "overall_risk_level": overall_verdict,
        "clauses": results,
        "missing_clauses": missing_clauses,
    }