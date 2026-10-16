def shorten_text(text: str, max_chars: int = 1200) -> str:
    if not text:
        return ""

    text = " ".join(str(text).split())

    if len(text) <= max_chars:
        return text

    return text[:max_chars].rstrip() + "..."


def risk_sort_value(risk: str) -> int:
    order = {
        "HIGH": 1,
        "MEDIUM": 2,
        "LOW": 3
    }

    return order.get(
        str(risk or "").upper(),
        4
    )


def build_clause_title(clause: dict) -> str:

    heading = clause.get("heading")
    category = clause.get(
        "category",
        "General Clause"
    )

    if heading and str(heading).lower() != "general clause":
        return str(heading).strip()

    return str(category).strip()


GENERIC_ACTIONS = (
    "clarify the scope, conditions, timelines",
    "review and negotiate this clause with appropriate legal guidance",
)


def _norm(text) -> str:
    return " ".join(str(text or "").lower().split())


def _sentence_clip(text: str, max_chars: int = 240) -> str:
    text = " ".join(str(text or "").split())
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    end = max(cut.rfind(". "), cut.rfind("; "))
    if end > max_chars * 0.5:
        return cut[:end + 1]
    return cut.rsplit(" ", 1)[0] + "..."


def _is_copy_of_original(safer: str, original: str) -> bool:
    """The MEDIUM-risk fallback just repeats the original clause as 'safer'."""
    a, b = _norm(safer)[:400], _norm(original)[:400]
    if not a or not b:
        return False
    if a == b or a[:120] in _norm(original) or b[:120] in _norm(safer):
        return True
    from difflib import SequenceMatcher
    return SequenceMatcher(None, a, b).quick_ratio() > 0.9


def build_negotiation_items(clauses: list, limit: int = 15) -> list:
    """One card per risky clause. Suggestions are clause-specific:
    topic advice, the LLM's own action (if not canned) and a real rewrite
    (if it differs from the original). Identical cards are dropped."""

    items = []
    seen = set()

    # actions repeated on 3+ clauses are canned text, not real advice
    counts = {}
    for c in clauses:
        key = _norm(c.get("action"))
        if key:
            counts[key] = counts.get(key, 0) + 1

    def is_generic(action: str) -> bool:
        key = _norm(action)
        return (
            not key
            or counts.get(key, 0) >= 3
            or any(g in key for g in GENERIC_ACTIONS)
        )

    sorted_clauses = sorted(
        clauses,
        key=lambda c: risk_sort_value(c.get("risk_level", "LOW"))
    )

    for clause in sorted_clauses:

        risk = str(clause.get("risk_level", "LOW")).upper()

        if risk not in ("HIGH", "MEDIUM"):
            continue

        issue = clause.get("detected_issue") or "Legal risk"
        action = clause.get("action") or ""
        advice = clause.get("negotiation_advice") or ""
        safer = clause.get("safer_clause") or ""
        original = clause.get("original_clause") or ""

        suggestions = []

        # 1) topic/issue advice first (always clause specific)
        if advice:
            suggestions.append(_sentence_clip(advice, 320))

        # 2) the LLM's own action, only if it is not canned text
        if action and not is_generic(action):
            text = _sentence_clip(action, 320)
            if _norm(text) not in [_norm(x) for x in suggestions]:
                suggestions.append(text)

        # 3) a real rewrite, only if it differs from the original clause
        if safer and not _is_copy_of_original(safer, original):
            suggestions.append("Suggested wording: " + _sentence_clip(safer, 240))

        suggestions = suggestions[:3]
        signature = tuple(_norm(x) for x in suggestions)

        if not suggestions or signature in seen:
            continue

        seen.add(signature)

        items.append({
            "problem": issue,
            "suggested_negotiation": suggestions,
            "priority": risk,
            "clause_no": clause.get("clause_id"),
            "clause_heading": shorten_text(clause.get("heading"), 80),
            "reason": shorten_text(clause.get("reason"), 300),
        })

        if len(items) >= limit:
            break

    return items


def build_simple_report(full_report: dict) -> dict:

    clauses = full_report.get(
        "clauses",
        []
    )

    sorted_clauses = sorted(
        clauses,
        key=lambda c: risk_sort_value(
            c.get("risk_level", "LOW")
        )
    )

    clause_analysis = []

    for clause in sorted_clauses:

        risk = str(
            clause.get(
                "risk_level",
                "LOW"
            )
        ).upper()

        # LOW risk clauses are not shown
        # in the detailed risk section.
        if risk == "LOW":
            continue

        clause_analysis.append({

            "clause_no":
                clause.get("clause_id"),

            "title":
                build_clause_title(clause),

            "risk_level":
                risk,

            "risk_score":
                clause.get("risk_score"),

            "category":
                clause.get("category"),

            "issue_identified":
                clause.get("detected_issue"),

            "original_clause":
                shorten_text(
                    clause.get(
                        "original_clause"
                    ),
                    1500
                ),

            "reason":
                shorten_text(
                    clause.get("reason"),
                    500
                ),

            "recommended_action":
                shorten_text(
                    clause.get("action"),
                    400
                ),

            "negotiation_suggestion":
                shorten_text(
                    clause.get(
                        "negotiation_advice"
                    ),
                    500
                ),

            "safer_clause":
                shorten_text(
                    clause.get(
                        "safer_clause"
                    ),
                    1500
                )
        })

    # =========================================================
    # MISSING CLAUSES
    # =========================================================

    missing_clauses = []

    for missing in full_report.get(
        "missing_clauses",
        []
    ):

        name = (
            missing.get("missing_clause")
            or missing.get("name")
            or missing.get("clause_name")
            or missing.get("title")
            or "Missing Clause"
        )

        reason = (
            missing.get("reason")
            or missing.get("why_important")
            or
            "This clause may be important "
            "for contractual protection."
        )

        risk_if_missing = (
            missing.get("risk_if_missing")
            or
            f"Absence of a {name} clause "
            "may create uncertainty or "
            "reduce legal protection."
        )

        suggested_clause = (
            missing.get("suggested_clause")
            or missing.get("recommended_clause")
            or ""
        )

        short_reason = shorten_text(
            reason,
            400
        )

        short_suggested_clause = shorten_text(
            suggested_clause,
            1200
        )

        missing_clauses.append({

            "name": name,

            "missing_clause": name,

            "why_important":
                short_reason,

            "reason":
                short_reason,

            "risk_if_missing":
                shorten_text(
                    risk_if_missing,
                    400
                ),

            "recommended_clause":
                short_suggested_clause,

            "suggested_clause":
                short_suggested_clause
        })

    # =========================================================
    # SUMMARY DATA
    # =========================================================

    summary = {

        "overall_risk_level":
            full_report.get(
                "overall_risk_level",
                "UNKNOWN"
            ),

        "overall_risk_score":
            full_report.get(
                "overall_risk_score",
                0
            ),

        "total_clauses_analyzed":
            full_report.get(
                "total_clauses",
                len(clauses)
            ),

        "high_risk_count":
            full_report.get(
                "high_risk",
                0
            ),

        "medium_risk_count":
            full_report.get(
                "medium_risk",
                0
            ),

        "low_risk_count":
            full_report.get(
                "low_risk",
                0
            ),

        "missing_clauses_count":
            len(missing_clauses),

        "important_clauses_count":
            len(clause_analysis)
    }

    # =========================================================
    # RISK DISTRIBUTION
    # =========================================================

    risk_distribution = {

        "high":
            summary["high_risk_count"],

        "medium":
            summary["medium_risk_count"],

        "low":
            summary["low_risk_count"]
    }

    # =========================================================
    # TRANSLATION
    # =========================================================

    translation = full_report.get(
        "translation",
        {}
    )

    return {

        "summary":
            summary,

        "risk_distribution":
            risk_distribution,

        "clause_analysis":
            clause_analysis,

        "missing_clauses":
            missing_clauses,

        "negotiation_assistant":
            build_negotiation_items(
                clauses
            ),

        # =====================================================
        # PROFESSIONAL DOCUMENT SUMMARY
        # =====================================================

        "document_summary": {

            "available":
                bool(
                    full_report.get(
                        "summary"
                    )
                ),

            "document_type":
                full_report.get(
                    "summary", {}
                ).get(
                    "document_type", "Unknown"
                ),

            "ai_summary":
                full_report.get(
                    "summary", {}
                ).get(
                    "ai_summary", ""
                ),

            "parties":
                full_report.get(
                    "summary", {}
                ).get(
                    "parties", []
                ),

            "important_dates":
                full_report.get(
                    "summary", {}
                ).get(
                    "important_dates", []
                ),

            "financial_terms":
                full_report.get(
                    "summary", {}
                ).get(
                    "financial_terms", []
                ),

            "key_terms_and_obligations":
                full_report.get(
                    "summary", {}
                ).get(
                    "key_terms_and_obligations", []
                ),

            "important_clauses":
                full_report.get(
                    "summary", {}
                ).get(
                    "important_clauses", []
                ),

            "key_legal_terms":
                full_report.get(
                    "summary", {}
                ).get(
                    "key_legal_terms", []
                ),

            "statistics":
                full_report.get(
                    "summary", {}
                ).get(
                    "statistics", {}
                ),

            "formatted_text":
                full_report.get(
                    "summary", {}
                ).get(
                    "formatted_text", ""
                ),

            "message":
                ""
                if full_report.get(
                    "summary"
                )
                else
                "Summary is not available."
        },

        # =====================================================
        # TRANSLATION
        # =====================================================

        "translation": {

            "available":
                bool(translation),

            "supported_languages": [
                "English",
                "Hindi",
                "Marathi"
            ],

            "english":
                translation.get(
                    "english",
                    ""
                ),

            "hindi":
                translation.get(
                    "hindi",
                    ""
                ),

            "marathi":
                translation.get(
                    "marathi",
                    ""
                )
        },

        # =====================================================
        # DOWNLOADS
        # =====================================================

        "downloads": {

            "json_report":
                full_report.get(
                    "report_path"
                ),

            "pdf_report":
                None,

            "docx_report":
                None
        }
    }


def simplify_clause_result(
    full_result: dict
) -> dict:

    return {

        "risk_level":
            full_result.get(
                "risk_level"
            ),

        "risk_score":
            full_result.get(
                "risk_score"
            ),

        "category":
            full_result.get(
                "category"
            ),

        "issue_identified":
            full_result.get(
                "detected_issue"
            ),

        "original_clause":
            shorten_text(
                full_result.get(
                    "original_clause"
                ),
                1500
            ),

        "reason":
            shorten_text(
                full_result.get(
                    "reason"
                ),
                500
            ),

        "recommended_action":
            shorten_text(
                full_result.get(
                    "action"
                ),
                400
            ),

        "negotiation_suggestion":
            shorten_text(
                full_result.get(
                    "negotiation_advice"
                ),
                500
            ),

        "safer_clause":
            shorten_text(
                full_result.get(
                    "safer_clause"
                ),
                1500
            )
    }