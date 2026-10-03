"""
LexiShield Missing Clause Detector

Purpose:
    Detect important missing or materially inadequate clauses
    using Groq LLM + local fallback rules.

Architecture:
    1. Detect document type
    2. Ask LLM for missing/inadequate clauses
    3. Safely extract JSON
    4. Normalize results
    5. Remove duplicates/already-present clauses
    6. Use local fallback if LLM fails

The LLM remains enabled.
"""

from openai import OpenAI
import json
import re
from services.llm_reasoner import is_gemini_available, mark_gemini_unavailable,call_with_transient_retry 
from services import llm_cache

from services.config import settings


# ============================================================
# GROQ CLIENT
# ============================================================

client = OpenAI(
    base_url=settings.LLM_BASE_URL,
    api_key=settings.LLM_API_KEY,
    timeout=settings.LLM_TIMEOUT,
    max_retries=0,
)

MODEL_NAME = settings.LLM_MODEL_NAME


# ============================================================
# BASIC HELPERS
# ============================================================

def _norm(text: str) -> str:
    """
    Normalize text for comparisons.
    """

    return " ".join(
        str(text or "")
        .lower()
        .split()
    )


def clean_missing_clause_text(
    text: str
) -> str:
    """
    Clean text returned by the LLM.
    """

    if not text:
        return ""

    cleaned = str(text).strip()

    # Remove markdown/code formatting.
    cleaned = cleaned.replace(
        "```json",
        ""
    )

    cleaned = cleaned.replace(
        "```",
        ""
    )

    # Normalize newlines.
    cleaned = cleaned.replace(
        "\r",
        " "
    )

    cleaned = cleaned.replace(
        "\n",
        " "
    )

    # Fix common OCR/LLM spacing problems.
    cleaned = re.sub(
        r"\s+",
        " ",
        cleaned
    )

    # Fix words accidentally joined together.
    cleaned = re.sub(
        r"(?<=[a-z])(?=[A-Z])",
        " ",
        cleaned
    )

    cleaned = re.sub(
        r"(?<=[a-zA-Z])(?=\d)",
        " ",
        cleaned
    )

    cleaned = re.sub(
        r"(?<=\d)(?=[A-Za-z])",
        " ",
        cleaned
    )

    cleaned = cleaned.strip()

    if (
        cleaned
        and cleaned[-1]
        not in ".!?"
    ):
        cleaned += "."

    return cleaned


# ============================================================
# JSON EXTRACTION
# ============================================================

def extract_missing_json(
    raw: str
):
    """
    Safely extract missing_clauses from an LLM response.

    Handles:
        - plain JSON
        - ```json ... ```
        - extra text before JSON
        - extra text after JSON
        - trailing commas
    """

    if not raw:
        return None

    raw = str(raw).strip()

    # --------------------------------------------------------
    # Remove markdown fences.
    # --------------------------------------------------------

    raw = re.sub(
        r"```json",
        "",
        raw,
        flags=re.IGNORECASE
    )

    raw = raw.replace(
        "```",
        ""
    ).strip()

    # --------------------------------------------------------
    # First attempt: complete JSON.
    # --------------------------------------------------------

    try:

        obj = json.loads(
            raw
        )

        if isinstance(obj, dict):

            clauses = obj.get(
                "missing_clauses"
            )

            if isinstance(
                clauses,
                list
            ):
                return clauses

    except Exception:
        pass

    # --------------------------------------------------------
    # Extract first JSON object.
    # --------------------------------------------------------

    start = raw.find(
        "{"
    )

    if start == -1:
        return None

    # Try progressively smaller ending positions.
    end_positions = [
        match.start()
        for match in re.finditer(
            r"}",
            raw
        )
    ]

    for end in reversed(
        end_positions
    ):

        candidate = raw[
            start:end + 1
        ].strip()

        # Remove trailing commas.
        candidate = re.sub(
            r",\s*([}\]])",
            r"\1",
            candidate
        )

        try:

            obj = json.loads(
                candidate
            )

            if not isinstance(
                obj,
                dict
            ):
                continue

            clauses = obj.get(
                "missing_clauses"
            )

            if isinstance(
                clauses,
                list
            ):
                return clauses

        except Exception:
            continue

    return None


# ============================================================
# DOCUMENT TYPE DETECTION
# ============================================================

def detect_document_type(
    full_text: str
) -> str:
    """
    Lightweight document type detection.
    """

    text = _norm(
        full_text
    )

    if any(
        x in text
        for x in [
            "employee",
            "employer",
            "employment agreement",
            "salary",
            "probation",
            "notice period"
        ]
    ):
        return "employment"

    if any(
        x in text
        for x in [
            "law firm",
            "legal services agreement",
            "attorney",
            "client's file",
            "lawyer-client privilege",
            "estate plan",
            "billing error"
        ]
    ):
        return "legal_services"

    if any(
        x in text
        for x in [
            "power of attorney",
            "principal",
            "attorney-in-fact",
            "agent",
            "execute sale deed"
        ]
    ):
        return "power_of_attorney"

    if any(
        x in text
        for x in [
            "non-disclosure",
            "nda",
            "confidential information",
            "receiving party",
            "disclosing party"
        ]
    ):
        return "nda"

    if any(
        x in text
        for x in [
            "lease",
            "landlord",
            "tenant",
            "premises",
            "rent",
            "security deposit"
        ]
    ):
        return "lease"

    if any(
        x in text
        for x in [
            "loan",
            "borrower",
            "lender",
            "interest rate",
            "default",
            "repayment"
        ]
    ):
        return "loan"

    if any(
        x in text
        for x in [
            "service provider",
            "customer",
            "vendor",
            "deliverables",
            "statement of work",
            "scope of services"
        ]
    ):
        return "service_agreement"

    return "general_contract"


# ============================================================
# DUPLICATE / EXISTING CLAUSE CHECK
# ============================================================

def has_substantially_similar_clause(
    full_text: str,
    missing_name: str
) -> bool:
    """
    Prevent clauses already present in the document
    from being reported as missing.
    """

    text = _norm(
        full_text
    )

    name = _norm(
        missing_name
    )

    checks = {

        "client termination": [
            "client may discharge law firm",
            "client may terminate",
            "discharge of law firm"
        ],

        "law firm withdrawal": [
            "withdrawal of law firm",
            "law firm may withdraw",
            "rules permit such withdrawal",
            "withdraw as client's attorney"
        ],

        "document retention": [
            "document retention",
            "retain client's file",
            "destroy all files",
            "storage facility",
            "client's file"
        ],

        "facsimile signature": [
            "facsimile signature",
            "facsimile signature on this agreement",
            "original signature"
        ],

        "client cooperation": [
            "client will be truthful and cooperative",
            "client will be truthful",
            "provide on a timely basis all information",
            "client keeping law firm fully informed"
        ],

        "scope of services": [
            "legal services to be provided",
            "services to be provided",
            "preparation of the following documents"
        ],

        "withdrawal": [
            "withdrawal of law firm",
            "law firm may withdraw",
            "rules permit such withdrawal"
        ],

        "termination": [
            "client may discharge law firm",
            "discharge of law firm",
            "effective when received by law firm"
        ],

        "file return": [
            "release of client's papers",
            "relinquish client's original documents",
            "client's original documents",
            "five working days written notice"
        ],

        "severability": [
            "severability",
            "partial invalidity",
            "unenforceable for any reason"
        ],

        "entire agreement": [
            "entire agreement",
            "complete agreement",
            "supersedes all previous"
        ],

        "modification": [
            "modified by subsequent agreement",
            "instrument in writing signed by both",
            "modified by written agreement"
        ],

        "signature": [
            "facsimile signature",
            "original signature",
            "signed copy"
        ],

        "effective date": [
            "effective date",
            "agreement will be the date"
        ]
    }

    for topic, patterns in checks.items():

        if topic in name:

            if any(
                pattern in text
                for pattern in patterns
            ):
                return True

    return False


# ============================================================
# INADEQUATE CLAUSE CHECK
# ============================================================

def should_convert_to_inadequate(
    full_text: str,
    missing_name: str
) -> bool:
    """
    Detect clauses that exist but appear weak/inadequate.
    """

    text = _norm(
        full_text
    )

    name = _norm(
        missing_name
    )

    if "confidential" in name:

        weak_confidentiality_signals = [

            "no information will be kept confidential",

            "neither of you will be able to invoke "
            "the lawyer-client privilege",

            "fully and freely disclosed to the other"
        ]

        if any(
            signal in text
            for signal in weak_confidentiality_signals
        ):
            return True

    if (
        "billing" in name
        or "invoice" in name
    ):

        weak_billing_signals = [

            "deemed to have been accepted",

            "without adjustment of any kind",

            "billing error or dispute within 30 days"
        ]

        if any(
            signal in text
            for signal in weak_billing_signals
        ):
            return True

    return False


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_missing_clause(
    item: dict
) -> dict:
    """
    Normalize LLM output to the expected schema.
    """

    if not isinstance(
        item,
        dict
    ):
        item = {}

    name = (
        item.get(
            "missing_clause"
        )
        or item.get(
            "name"
        )
        or item.get(
            "clause_name"
        )
        or item.get(
            "title"
        )
        or "Missing Important Clause"
    )

    reason = (
        item.get(
            "reason"
        )
        or item.get(
            "why_important"
        )
        or
        "This clause may be important "
        "for legal certainty and contractual protection."
    )

    suggested_clause = (
        item.get(
            "suggested_clause"
        )
        or item.get(
            "recommended_clause"
        )
        or
        "The parties shall include a clear "
        "and balanced clause addressing this issue "
        "in accordance with applicable law."
    )

    return {

        "missing_clause":
            clean_missing_clause_text(
                name
            ),

        "reason":
            clean_missing_clause_text(
                reason
            ),

        "suggested_clause":
            clean_missing_clause_text(
                suggested_clause
            )
    }


# ============================================================
# FILTERING
# ============================================================

def filter_missing_clauses(
    full_text: str,
    missing_clauses: list
) -> list:
    """
    Remove duplicates and clauses already present.
    """

    filtered = []

    seen = set()

    for item in missing_clauses:

        normalized = (
            normalize_missing_clause(
                item
            )
        )

        name = normalized[
            "missing_clause"
        ]

        key = _norm(
            name
        )

        if not key:
            continue

        if key in seen:
            continue

        # Existing but weak clause.
        if should_convert_to_inadequate(
            full_text,
            name
        ):

            if not key.startswith(
                "unclear or inadequate"
            ):

                normalized[
                    "missing_clause"
                ] = (
                    "Unclear or inadequate "
                    + name
                )

                normalized[
                    "reason"
                ] = (
                    "The document appears to contain "
                    "a related clause, but the wording "
                    "may be incomplete, weak, or "
                    "one-sided and should be clarified."
                )

            key = _norm(
                normalized[
                    "missing_clause"
                ]
            )

        # Already substantially present.
        elif has_substantially_similar_clause(
            full_text,
            name
        ):
            continue

        seen.add(
            key
        )

        filtered.append(
            normalized
        )

        # Maximum 8.
        if len(filtered) >= 8:
            break

    return filtered


# ============================================================
# LOCAL FALLBACK
# ============================================================

def fallback_missing_clauses(
    full_text: str
) -> list:
    """
    Reliable local fallback if LLM fails.
    """

    doc_type = detect_document_type(
        full_text
    )

    # --------------------------------------------------------
    # LEGAL SERVICES
    # --------------------------------------------------------

    if doc_type == "legal_services":

        items = [

            {
                "missing_clause":
                    "Unclear or inadequate confidentiality and privilege clause",

                "reason":
                    "Legal services may involve sensitive client information, and the agreement should clearly explain confidentiality and privilege treatment.",

                "suggested_clause":
                    "Law Firm shall maintain the confidentiality of Client information except as required by law, applicable professional conduct rules, or informed written consent of Client."
            },

            {
                "missing_clause":
                    "Limitation of Liability Clause",

                "reason":
                    "A legal services agreement should define reasonable limits for liability while preserving remedies for gross negligence, fraud, or wilful misconduct.",

                "suggested_clause":
                    "Law Firm shall not be liable for indirect or consequential losses, except to the extent caused by Law Firm's gross negligence, fraud, or wilful misconduct, and any limitation shall comply with applicable professional rules."
            },

            {
                "missing_clause":
                    "Governing Law and Jurisdiction Clause",

                "reason":
                    "The agreement should clearly state the law and forum that will govern disputes.",

                "suggested_clause":
                    "This Agreement shall be governed by the laws applicable to the jurisdiction where Law Firm provides the legal services, and disputes shall be resolved by courts or forums having competent jurisdiction."
            },

            {
                "missing_clause":
                    "Dispute Resolution Clause",

                "reason":
                    "A dispute resolution clause provides a clear process for resolving fee or service-related disputes.",

                "suggested_clause":
                    "The parties shall first attempt to resolve disputes in good faith, and unresolved disputes may be submitted to mediation, arbitration, fee dispute resolution, or a competent court as permitted by applicable law."
            },

            {
                "missing_clause":
                    "Clear billing dispute escalation process",

                "reason":
                    "The agreement should clearly explain how billing disputes will be raised, reviewed, and resolved.",

                "suggested_clause":
                    "Client may dispute any invoice in writing within a reasonable period, and Law Firm shall review the dispute in good faith before imposing collection costs, interest, or service charges on the disputed amount."
            },

            {
                "missing_clause":
                    "Data Protection and Privacy Clause",

                "reason":
                    "Legal services may involve sensitive personal and financial information, which should be protected.",

                "suggested_clause":
                    "Law Firm shall process and store Client personal information only for legitimate legal services purposes and shall apply reasonable administrative, technical, and organizational safeguards."
            }
        ]

        return filter_missing_clauses(
            full_text,
            items
        )

    # --------------------------------------------------------
    # EMPLOYMENT
    # --------------------------------------------------------

    if doc_type == "employment":

        items = [

            {
                "missing_clause":
                    "Clear job responsibilities and performance standards",

                "reason":
                    "Employment agreements should clearly define the employee's role and performance expectations.",

                "suggested_clause":
                    "The Employee's duties, reporting structure, performance standards, and working arrangements shall be clearly communicated in writing and applied fairly."
            },

            {
                "missing_clause":
                    "Governing Law and Jurisdiction Clause",

                "reason":
                    "The agreement should specify applicable law and forum for disputes.",

                "suggested_clause":
                    "This Agreement shall be governed by applicable Indian law, and disputes shall be subject to the jurisdiction of competent courts or forums as agreed by the parties."
            },

            {
                "missing_clause":
                    "Dispute Resolution Clause",

                "reason":
                    "Employment disputes should have a clear and fair resolution process.",

                "suggested_clause":
                    "The parties shall first attempt to resolve disputes amicably, and unresolved disputes may be referred to mediation, arbitration, labour authorities, or courts as permitted by law."
            },

            {
                "missing_clause":
                    "Data Protection and Privacy Clause",

                "reason":
                    "Employee personal information should be handled with appropriate safeguards.",

                "suggested_clause":
                    "Employer shall collect and process Employee personal data only for legitimate employment purposes and in accordance with applicable data protection requirements."
            }
        ]

        return filter_missing_clauses(
            full_text,
            items
        )

    # --------------------------------------------------------
    # POWER OF ATTORNEY
    # --------------------------------------------------------

    if doc_type == "power_of_attorney":

        items = [

            {
                "missing_clause":
                    "Revocation and Termination Clause",

                "reason":
                    "A power of attorney should clearly explain when and how authority ends.",

                "suggested_clause":
                    "This Power of Attorney may be revoked by the Principal by written notice and shall terminate upon completion of the specified purpose or as required by law."
            },

            {
                "missing_clause":
                    "Attorney-in-Fact Duties and Records Clause",

                "reason":
                    "The agent should be required to act in good faith and maintain transaction records.",

                "suggested_clause":
                    "The Attorney-in-Fact shall act in good faith, within the scope of authority granted, and maintain accurate records of all acts performed under this Power of Attorney."
            },

            {
                "missing_clause":
                    "Limitation of Authority Clause",

                "reason":
                    "Broad authority can create misuse risk unless limited.",

                "suggested_clause":
                    "The Attorney-in-Fact may perform only the specific acts expressly authorized in this instrument and shall not transfer or encumber property unless expressly permitted."
            },

            {
                "missing_clause":
                    "Governing Law and Jurisdiction Clause",

                "reason":
                    "The governing law and forum should be clear.",

                "suggested_clause":
                    "This Power of Attorney shall be governed by applicable law, and disputes shall be resolved before competent courts having jurisdiction."
            }
        ]

        return filter_missing_clauses(
            full_text,
            items
        )

    # --------------------------------------------------------
    # GENERAL CONTRACT
    # --------------------------------------------------------

    items = [

        {
            "missing_clause":
                "Governing Law and Jurisdiction Clause",

            "reason":
                "Most contracts should specify governing law and dispute forum.",

            "suggested_clause":
                "This Agreement shall be governed by applicable law, and disputes shall be subject to the jurisdiction of competent courts or agreed dispute resolution forums."
        },

        {
            "missing_clause":
                "Dispute Resolution Clause",

            "reason":
                "A dispute resolution mechanism reduces uncertainty if disputes arise.",

            "suggested_clause":
                "The parties shall first attempt good-faith resolution, and unresolved disputes may be submitted to mediation, arbitration, or competent courts as agreed by the parties."
        },

        {
            "missing_clause":
                "Confidentiality Clause",

            "reason":
                "Contracts often involve confidential information that should be protected.",

            "suggested_clause":
                "Each party shall protect confidential information received from the other and use it only for legitimate contractual purposes, subject to customary legal exceptions."
        },

        {
            "missing_clause":
                "Limitation of Liability Clause",

            "reason":
                "A liability clause helps define fair responsibility and reduce excessive exposure.",

            "suggested_clause":
                "Each party's liability shall be limited to direct losses caused by proven breach, negligence, fraud, or wilful misconduct, subject to reasonable limits permitted by law."
        }
    ]

    return filter_missing_clauses(
        full_text,
        items
    )


# ============================================================
# PROMPT
# ============================================================

def build_missing_clause_prompt(
    full_text: str
) -> str:
    """
    Compact prompt to reduce Groq token usage.
    """

    excerpt = str(
        full_text or ""
    )

    # Keep enough context for meaningful analysis.
    if len(excerpt) > 12000:
        excerpt = excerpt[:12000]

    doc_type = detect_document_type(
        full_text
    )

    prompt = f"""
Review this legal document for important clauses that are
truly missing or materially inadequate.

Document type:
{doc_type}

Document:
{excerpt}

Return ONLY valid JSON in exactly this format:

{{
  "missing_clauses": [
    {{
      "missing_clause": "...",
      "reason": "...",
      "suggested_clause": "..."
    }}
  ]
}}

Rules:
1. Maximum 8 items.
2. Do not report a clause that is already substantially present.
3. If a related clause exists but is materially weak, vague,
   incomplete, or one-sided, name it:
   "Unclear or inadequate [clause name]".
4. Focus on important legal protections.
5. Do not invent minor boilerplate.
6. Keep reason short.
7. Suggested clause must be complete and practical.
8. Return [] if no important missing/inadequate clause exists.
9. No markdown.
10. No explanation outside JSON.

Document-specific focus:

Employment:
termination, salary/wage protection, duties,
dispute resolution, governing law, privacy,
non-compete/non-solicitation fairness.

Legal services:
confidentiality/privilege, liability,
governing law, dispute/billing process,
privacy, file return.

Power of attorney:
revocation, agent duties, records,
authority limits, dispute resolution,
governing law.

Return ONLY JSON.
"""

    return prompt


# ============================================================
# MAIN LLM DETECTOR
# ============================================================
def detect_missing_clauses(
    full_text: str
) -> list:
    """
    Detect missing/inadequate clauses using
    Groq Structured Outputs.

    GPT-OSS-120B supports strict JSON Schema mode,
    so the LLM response is forced to follow the
    expected structure.
    """

    if not full_text:
        return []

    if not str(full_text).strip():
        return []

    prompt = build_missing_clause_prompt(
        full_text
    )
    cache_key = llm_cache.make_key("missing", MODEL_NAME, full_text)
    cached = llm_cache.get(cache_key)
    if cached is not None:
        print("[MissingClause] Served from cache (no Gemini call).")
        return cached

    if not is_gemini_available():
        print("[MissingClause] Gemini unavailable for this document. Using local fallback.")
        return fallback_missing_clauses(full_text)

    try:

        print(
            "[MissingClause] Calling LLM..."
        )

        response = call_with_transient_retry(lambda: client.chat.completions.create(

            model=MODEL_NAME,

            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict legal "
                        "missing-clause detector. "
                        "Analyze the document and "
                        "return only the requested "
                        "structured result."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature=0.0,

            max_tokens=3000,

            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "missing_clause_detection",
                    "strict": True,
                    "schema": {
                        "type": "object",

                        "properties": {

                            "missing_clauses": {
                                "type": "array",

                                "items": {
                                    "type": "object",

                                    "properties": {

                                        "missing_clause": {
                                            "type": "string"
                                        },

                                        "reason": {
                                            "type": "string"
                                        },

                                        "suggested_clause": {
                                            "type": "string"
                                        }
                                    },

                                    "required": [
                                        "missing_clause",
                                        "reason",
                                        "suggested_clause"
                                    ],

                                    "additionalProperties": False
                                }
                            }
                        },

                        "required": [
                            "missing_clauses"
                        ],

                        "additionalProperties": False
                    }
                }
            }
        ))

        raw = (
            response
            .choices[0]
            .message
            .content
        )

        print(
            "[MissingClause] "
            "LLM response received."
        )

        # ----------------------------------------------------
        # Parse guaranteed JSON.
        # ----------------------------------------------------

        data = json.loads(
            raw
        )

        parsed = data.get(
            "missing_clauses",
            []
        )

        if not isinstance(
            parsed,
            list
        ):
            raise ValueError(
                "missing_clauses is not a list"
            )

        # ----------------------------------------------------
        # Normalize.
        # ----------------------------------------------------

        normalized = []

        for item in parsed:

            if not isinstance(
                item,
                dict
            ):
                continue

            normalized.append(
                normalize_missing_clause(
                    item
                )
            )

        # ----------------------------------------------------
        # Remove duplicates / existing clauses.
        # ----------------------------------------------------

        filtered = filter_missing_clauses(
            full_text,
            normalized
        )

        if filtered:

            print(
                "[MissingClause] "
                f"Detected {len(filtered)} "
                "missing/inadequate clauses."
            )
            
            llm_cache.put_many([(cache_key, filtered)])
            return filtered

        # ----------------------------------------------------
        # Valid LLM response but nothing useful.
        # ----------------------------------------------------

        print(
            "[MissingClause] "
            "No useful LLM clauses. "
            "Using local fallback."
        )

        return fallback_missing_clauses(
            full_text
        )

    except Exception as e:
        mark_gemini_unavailable()

        print(
            "[MissingClause] "
            "LLM failed:"
        )

        print(
            str(e)[:300]
        )

        print(
            "[MissingClause] "
            "Using local fallback."
        )

        return fallback_missing_clauses(
            full_text
        )