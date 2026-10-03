"""
Validation-only calibration layer.

IMPORTANT (architecture decision):
Risk classification is decided ONLY by the LLM
(Gemini primary, local Qwen fallback -- see
semantic_risk_classifier.py). This module must never
upgrade/downgrade a risk level using keyword/pattern
matching, because that would make risk classification
partly rule-based again.

This module only:
    - validates risk_level is one of LOW/MEDIUM/HIGH/UNKNOWN
    - validates risk_score is in range (or None for UNKNOWN)
    - validates confidence is in range
    - fills in missing category/issue text safely

It never changes a risk_level that the LLM already produced.
"""

VALID_LEVELS = {"LOW", "MEDIUM", "HIGH", "UNKNOWN"}


def calibrate_risk(clause_text: str, risk_result: dict) -> dict:
    """
    Validate/normalize the LLM's risk result.

    Does NOT decide or change risk_level based on clause
    keywords or patterns. That decision belongs to the LLM.
    """

    if not isinstance(risk_result, dict):
        risk_result = {}

    calibrated = dict(risk_result)

    # --------------------------------------------------------
    # risk_level: validate only, never override
    # --------------------------------------------------------

    level = str(calibrated.get("risk_level", "UNKNOWN")).upper().strip()

    if level not in VALID_LEVELS:
        level = "UNKNOWN"

    calibrated["risk_level"] = level

    # --------------------------------------------------------
    # risk_score: validate range; UNKNOWN keeps score as None
    # --------------------------------------------------------

    if level == "UNKNOWN":
        calibrated["risk_score"] = calibrated.get("risk_score")
    else:
        try:
            score = int(float(calibrated.get("risk_score", 50)))
        except (TypeError, ValueError):
            score = 50

        calibrated["risk_score"] = max(0, min(score, 100))

    # --------------------------------------------------------
    # confidence: validate range
    # --------------------------------------------------------

    try:
        confidence = float(calibrated.get("confidence", 0.70))
    except (TypeError, ValueError):
        confidence = 0.70

    calibrated["confidence"] = max(0.0, min(confidence, 1.0))

    # --------------------------------------------------------
    # category / detected_issue: fill only if missing
    # --------------------------------------------------------

    if not str(calibrated.get("category", "")).strip():
        calibrated["category"] = "General Legal Clause"

    if not str(calibrated.get("detected_issue", "")).strip():
        calibrated["detected_issue"] = (
            "Unable to determine legal risk."
            if level == "UNKNOWN"
            else "No specific issue detail provided."
        )

    return calibrated
