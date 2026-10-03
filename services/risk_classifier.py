import re
from typing import Dict, List


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def _normalize_text(text: str) -> str:
    """
    Normalize text for reliable rule matching.
    """
    return " ".join(
        str(text or "")
        .lower()
        .split()
    )


def _contains_any(text: str, patterns: List[str]) -> bool:
    """
    Return True when any pattern exists in normalized text.
    """
    return any(
        pattern in text
        for pattern in patterns
    )


# ============================================================
# INPUT FILTER
# ============================================================

def is_non_legal_or_garbage(clause: str) -> bool:
    """
    Detect empty, extremely short, or OCR-garbage clauses.

    These clauses are not sent through the legal-risk rules.
    """

    if not clause:
        return True

    text = str(clause).strip()

    words = text.split()

    if len(words) < 6:
        return True

    alpha_count = sum(
        ch.isalpha()
        for ch in text
    )

    if len(text) > 20:

        alpha_ratio = (
            alpha_count / len(text)
        )

        if alpha_ratio < 0.30:
            return True

    return False


# ============================================================
# RISK LEVEL HELPERS
# ============================================================

def _risk_rank(level: str) -> int:
    """
    Convert risk level to comparable rank.
    """

    level = str(
        level or ""
    ).upper()

    if level == "HIGH":
        return 3

    if level == "MEDIUM":
        return 2

    return 1


def _score_for_level(level: str) -> int:
    """
    Standard local risk scores.
    """

    level = str(
        level or ""
    ).upper()

    if level == "HIGH":
        return 90

    if level == "MEDIUM":
        return 65

    return 20


# ============================================================
# LOCAL LEGAL RISK RULES
# ============================================================
#
# These rules are deliberately broad.
#
# Reason:
# A rule should help catch a genuinely dangerous clause even
# when a semantic model would otherwise classify it incorrectly.
#
# The calibration layer can further upgrade these results.
# ============================================================

HIGH_RISK_RULES = [

    {
        "category": "Termination / Suspension",
        "issue": "Termination or suspension without reasonable notice",
        "patterns": [
            "terminate without notice",
            "terminated without notice",
            "termination without notice",
            "without giving any notice",
            "without any notice",
            "terminate immediately without notice",
            "immediate termination",
            "suspend services immediately",
            "suspend immediately",
            "stop work on the file",
            "close client's file",
            "dismissed forthwith",
            "discharged forthwith",
            "withdraw immediately without notice",
            "withdraw without reasonable notice",
            "terminate at any time without notice"
        ]
    },

    {
        "category": "Termination",
        "issue": "Unilateral termination right",
        "patterns": [
            "terminate this agreement at its sole discretion",
            "terminate the agreement at its sole discretion",
            "terminate at its sole discretion",
            "termination at its sole discretion",
            "terminate for any reason",
            "terminate for any reason whatsoever",
            "may terminate at any time",
            "may terminate this agreement at any time"
        ]
    },

    {
        "category": "Payment Security",
        "issue": "Broad lien or payment security",
        "patterns": [
            "law firm shall have a lien",
            "lien for services rendered",
            "lien shall also cover",
            "any sums recovered",
            "settlement or judgment",
            "lien over any property",
            "lien over all amounts recovered"
        ]
    },

    {
        "category": "Payment",
        "issue": "Uncapped collection costs or service charge",
        "patterns": [
            "all costs of collection",
            "costs of collection plus",
            "service charge on any delinquent balance",
            "delinquent balance at the rate of",
            "pay all costs of collection",
            "all expenses of collection",
            "any and all collection costs"
        ]
    },

    {
        "category": "Penalty",
        "issue": "Excessive penalty or bond payment",
        "patterns": [
            "penalty",
            "rs. 2 lakhs",
            "rs 2 lakhs",
            "liable to pay",
            "break of bond",
            "breach of bond",
            "liquidated damages without proof",
            "pay damages for leaving",
            "penalty for resignation",
            "penalty for termination"
        ]
    },

    {
        "category": "Payment",
        "issue": "Unfair salary or payment withholding",
        "patterns": [
            "not entitled to salary",
            "withhold salary",
            "salary shall be withheld",
            "payment shall be withheld",
            "withholding of salary",
            "withhold any payment",
            "salary of the notice period",
            "shall not be paid",
            "payment may be withheld"
        ]
    },

    {
        "category": "Liability",
        "issue": "Uncapped or unlimited liability",
        "patterns": [
            "unlimited liability",
            "without limitation",
            "all losses",
            "all damages",
            "indirect losses",
            "remote losses",
            "consequential damages without limitation",
            "liable for any and all losses",
            "liable for all damages",
            "any and all damages"
        ]
    },

    {
        "category": "Indemnity",
        "issue": "Broad or uncapped indemnity",
        "patterns": [
            "indemnify and keep indemnified",
            "all claims, losses, damages",
            "whether direct or indirect",
            "without any limit",
            "all expenses",
            "any and all claims",
            "any and all losses",
            "any and all damages",
            "indemnify against all"
        ]
    },

    {
        "category": "Amendment",
        "issue": "Unilateral modification",
        "patterns": [
            "modify the agreement",
            "modify this agreement",
            "modify policies from time to time",
            "changes shall apply immediately",
            "applicable from the same day",
            "at its sole discretion",
            "may modify the terms",
            "may change the terms",
            "change the agreement without",
            "amend the agreement without"
        ]
    },

    {
        "category": "Non-Compete",
        "issue": "Overbroad non-compete",
        "patterns": [
            "not open similar",
            "non compete",
            "non-compete",
            "similar nature company",
            "three years directly or indirectly",
            "five years worldwide",
            "worldwide non-compete",
            "for a period of three years",
            "for a period of five years",
            "shall not engage in any competing business"
        ]
    },

    {
        "category": "Intellectual Property",
        "issue": "Overbroad intellectual property ownership",
        "patterns": [
            "all ideas",
            "all inventions",
            "created at any time",
            "exclusive property of the company",
            "whether or not related to company business",
            "source code shall remain exclusive property",
            "all intellectual property created",
            "all intellectual property shall belong",
            "any invention made during"
        ]
    },

    {
        "category": "Power of Attorney",
        "issue": "Broad property authority",
        "patterns": [
            "attorney-in-fact may sell",
            "attorney-in-fact may transfer",
            "attorney-in-fact may mortgage",
            "authorized to sell",
            "authorized to transfer",
            "authorized to mortgage",
            "sell, transfer and mortgage",
            "execute sale deed",
            "execute any sale deed",
            "all acts, deeds and things",
            "irrevocable power of attorney",
            "power of attorney is irrevocable",
            "shall remain valid indefinitely",
            "dispose of the property",
            "gift the property"
        ]
    },

    {
        "category": "Dispute Resolution",
        "issue": "One-sided dispute decision or waiver of rights",
        "patterns": [
            "final and binding on the employee",
            "decision of management",
            "decision of the board will be final",
            "decision of management shall be final",
            "waives the right to approach",
            "no right to sue",
            "waive any right to claim",
            "waives all rights",
            "waive the right to legal action",
            "sole arbitrator appointed by"
        ]
    },

    {
        "category": "Data Protection",
        "issue": "Broad personal data use",
        "patterns": [
            "personal data may be used for any purpose",
            "personal information may be used for any purpose",
            "share personal data without consent",
            "transfer personal data without consent",
            "process personal data for any purpose",
            "use personal data for any purpose",
            "personal data may be used, shared, transferred, or stored",
            "share personal information with any third party",
            "transfer personal information to any third party"
        ]
    },

    {
        "category": "Privacy / Surveillance",
        "issue": "Broad employee surveillance or biometric monitoring",
        "patterns": [
            "biometric monitoring",
            "biometric data",
            "monitor all communications",
            "monitor all activities",
            "surveillance at all times",
            "continuous monitoring",
            "monitor personal devices",
            "access employee devices",
            "track location at all times"
        ]
    },

    {
        "category": "Warranty",
        "issue": "Broad warranty disclaimer",
        "patterns": [
            "no warranty of any kind",
            "disclaims all responsibility",
            "as is",
            "without any warranty",
            "all warranties are disclaimed",
            "without representation or warranty"
        ]
    },

    {
        "category": "Loan",
        "issue": "Acceleration without notice or cure period",
        "patterns": [
            "entire loan immediately due",
            "declare the entire loan immediately due",
            "loan immediately due and payable",
            "accelerate the loan without notice",
            "acceleration without cure period",
            "declare the entire loan without cure period",
            "entire outstanding amount immediately due"
        ]
    },

    {
        "category": "Lease",
        "issue": "Unrestricted entry or unilateral rent increase",
        "patterns": [
            "landlord may enter the premises at any time",
            "landlord may enter premises at any time",
            "landlord may enter without prior notice",
            "landlord may increase rent at any time",
            "rent may be increased at any time",
            "rent at its sole discretion",
            "increase rent at its sole discretion"
        ]
    },

    {
        "category": "Rights Waiver",
        "issue": "Broad waiver of legal rights",
        "patterns": [
            "waives all claims",
            "waive all claims",
            "waives any and all claims",
            "release all claims",
            "release and discharge from all claims",
            "waiver of all legal rights",
            "waive any legal remedy",
            "waive the right to claim"
        ]
    },

    {
        "category": "Confidentiality / Privilege",
        "issue": "Privilege or confidentiality waiver",
        "patterns": [
            "no information will be kept confidential",
            "lawyer-client privilege",
            "unable to invoke",
            "compel us to testify",
            "fully and freely disclosed",
            "waive attorney client privilege",
            "waiver of confidentiality"
        ]
    }
]


MEDIUM_RISK_RULES = [

    {
        "category": "Payment",
        "issue": "Deemed acceptance of disputed charges",
        "patterns": [
            "deemed to have been accepted",
            "billing error or dispute within 30 days",
            "pay for such charges in full",
            "without adjustment of any kind"
        ]
    },

    {
        "category": "Conflict of Interest",
        "issue": "Potential representation conflict",
        "patterns": [
            "conflict of interest",
            "dual representation",
            "represent both of you",
            "separate counsel",
            "joint representation"
        ]
    },

    {
        "category": "Document Retention",
        "issue": "File destruction or retrieval restriction",
        "patterns": [
            "destroy all files",
            "without notifying client",
            "storage facility",
            "retain client's file",
            "charge a $25.00 fee",
            "file retrieval fee"
        ]
    },

    {
        "category": "Fees",
        "issue": "Unilateral rate change or fee increase",
        "patterns": [
            "rates on this schedule are subject to change",
            "subject to change on 30 days written notice",
            "decline to pay any increased rates",
            "increased rates",
            "fees may be changed",
            "rates may be changed"
        ]
    },

    {
        "category": "Payment Liability",
        "issue": "Joint and several payment liability",
        "patterns": [
            "jointly and severally",
            "liable jointly and severally"
        ]
    },

    {
        "category": "Non-Solicitation",
        "issue": "Broad client or customer restriction",
        "patterns": [
            "client of the company",
            "customer of the company",
            "directly or indirectly",
            "shall not contact any client",
            "shall not solicit",
            "shall not solicit any customer"
        ]
    },

    {
        "category": "Renewal",
        "issue": "Auto-renewal or long cancellation window",
        "patterns": [
            "automatically renew",
            "successive terms",
            "cancelled at least ninety days",
            "auto-renewal",
            "automatically renewed"
        ]
    },

    {
        "category": "Assignment",
        "issue": "Assignment without clear consent",
        "patterns": [
            "assign this agreement",
            "without consent",
            "transfer its rights and obligations"
        ]
    },

    {
        "category": "Confidentiality",
        "issue": "Broad confidentiality obligation",
        "patterns": [
            "during and after",
            "confidential information",
            "confidentiality obligations",
            "remain confidential indefinitely",
            "indefinitely confidential"
        ]
    },

    {
        "category": "Discretion",
        "issue": "Broad discretionary power",
        "patterns": [
            "sole discretion",
            "absolute discretion",
            "as decided by",
            "as determined by",
            "in its discretion"
        ]
    },

    {
        "category": "Obligation",
        "issue": "Ambiguous or potentially broad obligation",
        "patterns": [
            "best efforts",
            "reasonable efforts",
            "as necessary",
            "as appropriate",
            "from time to time",
            "as may be required"
        ]
    },

    {
        "category": "Policy",
        "issue": "Unclear policy change process",
        "patterns": [
            "subject to company policy",
            "policies as amended from time to time",
            "policies may change",
            "rules may be changed"
        ]
    }
]


# ============================================================
# STANDARD LOW-RISK PATTERNS
# ============================================================

STANDARD_LOW_PATTERNS = [

    "identification of parties",
    "severability",
    "entire agreement",
    "facsimile signature",
    "effective date",
    "errors and omissions insurance",
    "no guarantee of outcome",
    "disclaimer of guaranty",
    "notice addresses",
    "definitions",
    "interpretation",
    "counterparts",
    "headings",
    "governing law",
]


# ============================================================
# CONTEXTUAL NEGATION CHECK
# ============================================================

NEGATION_WORDS = [
    "not",
    "no",
    "never",
    "shall not",
    "will not",
    "does not",
    "do not",
    "without",
]


def _pattern_is_negated(
    text: str,
    pattern: str
) -> bool:
    """
    Basic protection against obvious false positives.

    Example:
        "The company shall NOT terminate without notice."

    We do not want the phrase
        "terminate without notice"
    alone to automatically make the clause HIGH.
    """

    index = text.find(pattern)

    if index < 0:
        return False

    start = max(
        0,
        index - 60
    )

    context = text[start:index].strip()

    for negation in NEGATION_WORDS:

        if context.endswith(negation):
            return True

    return False


# ============================================================
# RULE MATCHING
# ============================================================

def _find_high_risk_rule(
    text: str
) -> dict | None:
    """
    Find the first strong HIGH-risk rule.

    Strong legal-risk patterns have priority over generic
    MEDIUM patterns.
    """

    for rule in HIGH_RISK_RULES:

        for pattern in rule["patterns"]:

            if pattern in text:

                if _pattern_is_negated(
                    text,
                    pattern
                ):
                    continue

                return rule

    return None


def _find_medium_risk_rule(
    text: str
) -> dict | None:
    """
    Find a MEDIUM-risk rule.
    """

    for rule in MEDIUM_RISK_RULES:

        for pattern in rule["patterns"]:

            if pattern in text:

                if _pattern_is_negated(
                    text,
                    pattern
                ):
                    continue

                return rule

    return None


# ============================================================
# LOCAL CLASSIFICATION
# ============================================================

def classify_risk(
    clause: str
) -> dict:
    """
    Local legal-risk classifier.

    IMPORTANT:
    This function does NOT call Gemini or any external LLM.

    Classification order:

        1. Input validation
        2. Strong HIGH-risk legal rules
        3. MEDIUM-risk legal rules
        4. Standard LOW patterns
        5. Conservative LOW result

    The legal_risk_calibrator.py layer can further upgrade
    risky clauses after this function.
    """

    if is_non_legal_or_garbage(
        clause
    ):

        return {
            "risk_level": "LOW",
            "risk_score": 0,
            "category": "Non-Legal Text",
            "detected_issue": "Not enough legal content",
            "classifier": "local_input_filter",
            "confidence": 0.99
        }

    text = _normalize_text(
        clause
    )

    # --------------------------------------------------------
    # HIGH
    # --------------------------------------------------------

    high_rule = _find_high_risk_rule(
        text
    )

    if high_rule is not None:

        return {
            "risk_level": "HIGH",
            "risk_score": 90,
            "category": high_rule["category"],
            "detected_issue": high_rule["issue"],
            "classifier": "local_legal_rule",
            "confidence": 0.90
        }

    # --------------------------------------------------------
    # MEDIUM
    # --------------------------------------------------------

    medium_rule = _find_medium_risk_rule(
        text
    )

    if medium_rule is not None:

        return {
            "risk_level": "MEDIUM",
            "risk_score": 65,
            "category": medium_rule["category"],
            "detected_issue": medium_rule["issue"],
            "classifier": "local_legal_rule",
            "confidence": 0.78
        }

    # --------------------------------------------------------
    # STANDARD LOW
    # --------------------------------------------------------

    if _contains_any(
        text,
        STANDARD_LOW_PATTERNS
    ):

        return {
            "risk_level": "LOW",
            "risk_score": 20,
            "category": "Standard Contract Clause",
            "detected_issue": "Standard legal boilerplate",
            "classifier": "local_standard_rule",
            "confidence": 0.88
        }

    # --------------------------------------------------------
    # DEFAULT
    # --------------------------------------------------------
    #
    # Do NOT pretend that an unknown clause is highly risky.
    # The separate semantic-risk layer will later review
    # ambiguous clauses.
    #
    # Also avoid an artificially high confidence.
    # --------------------------------------------------------

    return {
        "risk_level": "LOW",
        "risk_score": 20,
        "category": "General Legal Clause",
        "detected_issue": "No strong local risk pattern detected",
        "classifier": "local_rule_fallback",
        "confidence": 0.55
    }


# ============================================================
# LOCAL BATCH CLASSIFICATION
# ============================================================

def classify_semantic_risk_batch(
    clauses,
    default_category="General Legal Clause",
    batch_size=10
):
    """
    Compatibility batch function.

    IMPORTANT:
    This function intentionally does NOT call an LLM.

    It keeps the same public function name so existing code
    does not immediately break.

    Each clause is classified independently.

    batch_size is retained for API compatibility and future
    semantic batching, but no API request is made here.
    """

    results = []

    if not clauses:
        return results

    for start in range(
        0,
        len(clauses),
        batch_size
    ):

        batch = clauses[
            start:start + batch_size
        ]

        for clause in batch:

            result = classify_risk(
                clause
            )

            if (
                result.get("category")
                == "General Legal Clause"
                and default_category
            ):

                result["category"] = (
                    default_category
                )

            results.append(
                result
            )

    return results


# ============================================================
# SINGLE-CLAUSE SEMANTIC COMPATIBILITY FUNCTION
# ============================================================

def classify_semantic_risk(
    clause: str,
    default_category="General Legal Clause"
) -> dict:
    """
    Compatibility wrapper.

    Older modules may import classify_semantic_risk()
    from this file.

    It now performs local classification only.
    """

    result = classify_risk(
        clause
    )

    if (
        result.get("category")
        == "General Legal Clause"
        and default_category
    ):

        result["category"] = (
            default_category
        )

    return result