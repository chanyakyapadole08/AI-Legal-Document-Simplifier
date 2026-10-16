from openai import OpenAI
import json
import re
import time

from services.config import settings
from services import local_llm
from services import llm_cache

# ============================================================
# LLM CLIENT
# ============================================================

client = OpenAI(
    base_url=settings.LLM_BASE_URL,
    api_key=settings.LLM_API_KEY,
    timeout=settings.LLM_TIMEOUT,
    max_retries=0,
)

MODEL_NAME = settings.LLM_MODEL_NAME

# Sticky fallback: once Gemini fails once, skip it for the
# rest of the current document (reset per PDF via reset_gemini_availability()).
_gemini_available = True


def reset_gemini_availability():
    global _gemini_available
    _gemini_available = True


# ============================================================
# JSON EXTRACTION
# ============================================================

def extract_json(raw: str):
    """
    Extract a JSON object from an LLM response.

    Handles:
    - Normal JSON
    - ```json ... ```
    - Extra text before/after JSON
    - Nested JSON objects
    """

    if not raw:
        return None

    raw = str(raw).strip()

    if not raw:
        return None

    # --------------------------------------------------------
    # Remove markdown fences
    # --------------------------------------------------------

    raw = re.sub(r"```json\s*", "", raw, flags=re.IGNORECASE)
    raw = raw.replace("```", "")
    raw = raw.strip()

    # --------------------------------------------------------
    # First attempt: entire response
    # --------------------------------------------------------

    try:
        result = json.loads(raw)
        if isinstance(result, dict):
            return result
    except Exception:
        pass

    # --------------------------------------------------------
    # Find first JSON object using brace matching
    # --------------------------------------------------------

    start = raw.find("{")

    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False

    for index in range(start, len(raw)):
        char = raw[index]

        # Handle escaped characters inside JSON strings
        if escape:
            escape = False
            continue

        if char == "\\" and in_string:
            escape = True
            continue

        # Toggle string state
        if char == '"':
            in_string = not in_string
            continue

        # Ignore braces inside strings
        if in_string:
            continue

        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1

            if depth == 0:
                json_text = raw[start:index + 1]

                try:
                    result = json.loads(json_text)
                    if isinstance(result, dict):
                        return result
                except Exception:
                    pass

                break

    return None


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_generated_text(text: str) -> str:
    """
    Clean minor formatting problems in generated legal text.
    """

    if not text:
        return ""

    cleaned = str(text).strip()

    # --------------------------------------------------------
    # Safe replacements
    #
    # IMPORTANT:
    # Do not use "rs.1" because it can affect rs.10, rs.100 etc.
    # --------------------------------------------------------

    replacements = {
        "allowsthe": "allows the",
        "canlead": "can lead",
        "orbond": "or bond",
        "systemwith": "system with",
        "apenalty": "a penalty",
        "rs.75": "Rs. 75",
        "rs.50": "Rs. 50",
        "otherwiserequired": "otherwise required",
        "shallrequire": "shall require",
        "calculatedbased": "calculated based",
        "withoutany": "without any",
        "dueto": "due to",
        "mayterminate": "may terminate",
        "caseno": "case no",
        "uponproviding": "upon providing",
        "priorto": "prior to",
        "writtennotice": "written notice",
        "withoutproviding": "without providing",
        "inwhich": "in which",
        "theemployee": "the employee",
        "employment of theemployee": "employment of the employee",
        "andneither": "and neither",
        "informationand": "information and",
        "withoutthe": "without the",
        "thecompany": "the company",
        "employmentterms": "employment terms",
        "agreementthat": "agreement that",
        "interestsof": "interests of",
        "employmentlaws": "employment laws",
        "theemployer": "the employer",
        "broadproperty": "broad property",
        "legalrights": "legal rights",
    }

    for wrong, correct in replacements.items():
        cleaned = cleaned.replace(wrong, correct)

    # --------------------------------------------------------
    # Normalize Rs. spacing safely
    # --------------------------------------------------------

    cleaned = re.sub(r"\bRs\.\s*", "Rs. ", cleaned, flags=re.IGNORECASE)

    # --------------------------------------------------------
    # Remove unnecessary newlines
    # --------------------------------------------------------

    cleaned = cleaned.replace("\n", " ")
    cleaned = " ".join(cleaned.split())

    # --------------------------------------------------------
    # Add punctuation if missing
    # --------------------------------------------------------

    if cleaned and cleaned[-1] not in ".!?":
        cleaned += "."

    return cleaned


# ============================================================
# CLAUSE CONTEXT
# ============================================================

def infer_clause_context(clause: str, risk_result: dict) -> str:
    """
    Infer the document/clause context.
    """

    text = str(clause or "").lower()
    category = str(risk_result.get("category", "")).lower()
    combined = f"{text} {category}"

    # --------------------------------------------------------
    # Employment
    # --------------------------------------------------------

    if any(
        word in combined
        for word in [
            "employee", "employer", "employment", "salary",
            "notice period", "probation", "non-compete", "misconduct",
        ]
    ):
        return "employment"

    # --------------------------------------------------------
    # Legal Services
    # --------------------------------------------------------

    if any(
        word in combined
        for word in [
            "law firm", "client", "attorney", "legal services",
            "lawyer-client privilege", "joint representation",
            "estate plan", "billing", "lien for services", "client's file",
        ]
    ):
        return "legal_services"

    # --------------------------------------------------------
    # Power of Attorney
    # --------------------------------------------------------

    if any(
        word in combined
        for word in [
            "power of attorney", "attorney-in-fact", "principal",
            "agent", "sale deed", "mortgage the property",
        ]
    ):
        return "power_of_attorney"

    # --------------------------------------------------------
    # Lease
    # --------------------------------------------------------

    if any(
        word in combined
        for word in [
            "landlord", "tenant", "lease", "premises", "rent",
            "security deposit",
        ]
    ):
        return "lease"

    # --------------------------------------------------------
    # Loan
    # --------------------------------------------------------

    if any(
        word in combined
        for word in [
            "borrower", "lender", "loan", "interest", "default",
            "acceleration",
        ]
    ):
        return "loan"

    # --------------------------------------------------------
    # Service Agreement
    # --------------------------------------------------------

    if any(
        word in combined
        for word in [
            "service provider", "customer", "vendor", "deliverables",
            "statement of work", "scope of services",
        ]
    ):
        return "service_agreement"

    return "general_contract"


# ============================================================
# SAFER CLAUSE CONTEXT VALIDATION
# ============================================================

def safer_clause_context_mismatch(original_clause: str, safer_clause: str, context: str) -> bool:
    """
    Detect employment-specific language
    in non-employment clauses.
    """

    if not safer_clause:
        return True

    safer = str(safer_clause).lower()

    employment_terms = [
        "employee", "employer", "employment", "salary",
        "benefits", "accrued dues", "gross misconduct",
    ]

    if context != "employment":
        if any(term in safer for term in employment_terms):
            return True

    return False


# ============================================================
# LLM PROMPT
# ============================================================

def build_reasoning_prompt(clause: str, risk_result: dict, similar_clauses=None) -> str:
    """
    Build strict legal reasoning prompt.
    """

    if similar_clauses is None:
        similar_clauses = []

    context = infer_clause_context(clause, risk_result)

    similar_text = ""

    for item in similar_clauses[:3]:
        if not isinstance(item, dict):
            continue

        clause_text = (
            item.get("clause_text")
            or item.get("clause")
            or item.get("text")
            or ""
        )

        clause_type = item.get("clause_type", "Unknown")
        risk_label = item.get("risk_label", "Unknown")
        source = item.get("source", "Unknown")
        similarity = item.get("similarity_score", "")
        act_name = item.get("act_name", "")
        section_reference = item.get("section_reference", "")

        similar_text += (
            f"- Source: {source}; "
            f"Act: {act_name}; "
            f"Section/Reference: {section_reference}; "
            f"Clause Type: {clause_type}; "
            f"Risk Label: {risk_label}; "
            f"Similarity: {similarity}\n"
            f"  Text: {clause_text[:500]}\n"
        )

    if not similar_text:
        similar_text = "No similar clauses available."

    prompt = f"""
You are a professional legal contract review assistant.

You must return ONLY valid JSON.

Do not write markdown.
Do not write code fences.
Do not write explanation outside JSON.

Do not change the classifier risk level.

Do not use placeholders such as:
[X], [days], [amount], [party], [jurisdiction].

The classifier has already decided the risk level.

Document Context:
{context}

Classifier Output:
- Risk Level: {risk_result.get("risk_level")}
- Category: {risk_result.get("category")}
- Detected Issue: {risk_result.get("detected_issue")}

Original Clause:
{clause}

Retrieved Similar Legal Clauses and Legal References:
{similar_text}

Your task:

1. Provide exactly one professional legal reason.
2. Provide exactly one recommended action.
3. Provide exactly one safer rewritten clause.

IMPORTANT:

Do NOT generate negotiation advice.

Negotiation advice is handled separately by the application's
negotiation engine.

Context rules:

- Match the safer clause to the actual document context.
- Do not assume the document is an employment agreement unless
  the original clause clearly contains employment terms.
- If context is legal_services, use party labels such as
  "Law Firm" and "Client".
- If context is service_agreement, use "Service Provider"
  and "Customer" or neutral "parties".
- If context is power_of_attorney, use "Principal" and
  "Attorney-in-Fact" or "Agent".
- If context is lease, use "Landlord" and "Tenant".
- If context is loan, use "Lender" and "Borrower".
- If context is general_contract, use neutral "party" or "parties".
- Never generate employment-specific remedies such as salary,
  employee benefits, accrued dues, probation, or gross misconduct
  unless the context is employment.

Return exactly this JSON structure:

{{
  "reason": "one clear professional legal reason",
  "action": "one specific recommended action",
  "safer_clause": "one complete safer contract clause"
}}

General rules:

- Do not provide multiple reasons.
- Do not provide multiple actions.
- Do not provide multiple safer clause alternatives.
- Do not use bullet points inside JSON values.
- Keep every field as a single paragraph string.
- Use retrieved similar clauses and legal references only as reference.
- Do not copy retrieved clauses blindly.
- If retrieved clauses have UNKNOWN risk_label, treat them only as context.
- Prefer risk-labelled examples and official legal references when available.

Rules for safer_clause:

- Generate exactly one final safer_clause.
- The safer_clause must be one complete formal contract clause ready
  to insert into the agreement.
- It should be legally strong and balanced.
- Preserve the original commercial purpose.
- Do not invent or increase monetary amounts, penalties, fees,
  or compensation.
- If the issue is employee penalty, bond penalty, or liquidated
  damages, do not invent new fixed amounts or slabs.
- Prefer wording that limits recovery to reasonable, documented,
  legally recoverable actual costs.
- Include prior written notice and reasonable opportunity to respond
  where appropriate.
- If the issue is termination or service suspension, include
  reasonable prior written notice and cure opportunity where appropriate.
- For employment termination without notice, include protections
  appropriate to employment context.
- For legal services withdrawal or file closure, protect client
  interests and comply with professional obligations.
- For broad lien or payment security, limit the lien to lawful
  undisputed amounts and reasonable documented costs.
- For unlimited liability, add a reasonable liability cap.
- For uncapped indemnity, make the indemnity mutual and subject
  to a reasonable cap.
- For unilateral amendment or rate change, require prior written
  notice and mutual written consent where appropriate.
- For broad or ambiguous legal wording, use clear objective standards.
- For confidentiality clauses, use a reasonable duration except
  for legally protected trade secrets.
- For post-employment non-compete clauses, limit restrictions by
  time, geography, and scope.
"""

    return prompt


# ============================================================
# DEFAULT SAFER CLAUSE
# ============================================================

def get_default_safer_clause(risk_result: dict, clause: str = "") -> str:
    """
    Strong fallback safer clause based on issue and context.
    """

    issue = str(risk_result.get("detected_issue", "")).lower()
    category = str(risk_result.get("category", "")).lower()
    context = infer_clause_context(clause, risk_result)
    combined = f"{issue} {category}"

    # --------------------------------------------------------
    # PENALTY / BOND / LIQUIDATED DAMAGES
    # Context-aware
    # --------------------------------------------------------

    if "penalty" in combined or "bond" in combined or "liquidated damages" in combined:

        if context == "employment":
            return (
                "The Employee shall not be required to pay a fixed "
                "penalty merely because employment ends before a "
                "specified period. Any recovery by the Employer shall "
                "be limited to reasonable, documented, and legally "
                "recoverable costs actually incurred for training or "
                "onboarding, if any, and only after providing the "
                "Employee with prior written notice, supporting details, "
                "and a reasonable opportunity to respond."
            )

        if context == "lease":
            return (
                "Any penalty or charge arising from a breach of this "
                "Agreement shall be limited to reasonable, documented, "
                "and legally recoverable costs actually incurred by "
                "the Landlord, after prior written notice describing "
                "the breach and a reasonable opportunity for the Tenant "
                "to cure the breach where cure is reasonably possible."
            )

        if context == "loan":
            return (
                "Any penalty, default charge, or liquidated damages "
                "arising under this Agreement shall be limited to "
                "reasonable, documented, and legally recoverable "
                "amounts, shall be proportionate to the actual breach "
                "or loss, and shall be imposed only after written notice "
                "of the default and a reasonable opportunity to cure "
                "where permitted by applicable law."
            )

        if context == "legal_services":
            return (
                "Any penalty or additional charge payable under this "
                "Agreement shall be limited to reasonable, documented, "
                "and legally recoverable costs actually incurred, "
                "after written notice identifying the basis of the "
                "charge and a reasonable opportunity for Client to "
                "resolve or dispute the amount in good faith."
            )

        if context == "service_agreement":
            return (
                "Any penalty or liquidated damages payable under this "
                "Agreement shall be reasonable and proportionate to "
                "the proven contractual breach, supported by documented "
                "costs or loss where applicable, and imposed only after "
                "written notice and a reasonable opportunity to cure "
                "where cure is reasonably possible."
            )

        return (
            "Any penalty, bond recovery, or liquidated damages under "
            "this Agreement shall be limited to reasonable, documented, "
            "and legally recoverable amounts proportionate to the "
            "actual contractual breach, following prior written notice "
            "and a reasonable opportunity to cure where appropriate."
        )

    # --------------------------------------------------------
    # LEGAL SERVICES
    # --------------------------------------------------------

    if context == "legal_services":

        if (
            "termination" in combined
            or "suspension" in combined
            or "withdraw" in combined
            or "file closure" in combined
        ):
            return (
                "Law Firm may suspend non-urgent work or withdraw "
                "from the engagement only after providing Client "
                "with reasonable prior written notice, a reasonable "
                "opportunity to cure any non-payment or remediable "
                "breach, and sufficient time to protect Client's "
                "interests, subject at all times to applicable "
                "professional conduct obligations. Upon withdrawal "
                "or file closure, Law Firm shall cooperate reasonably "
                "in transferring the matter and returning Client "
                "papers and property, while preserving any lawful "
                "rights to collect undisputed earned fees and "
                "reasonable costs."
            )

        if "lien" in combined or "payment security" in combined:
            return (
                "Any lien or payment security claimed by Law Firm "
                "shall be limited to undisputed earned fees and "
                "reasonable documented costs, shall apply only to "
                "the extent permitted by applicable professional "
                "conduct rules, and shall not prevent Client from "
                "disputing charges in good faith or obtaining Client "
                "papers and property required to protect Client's "
                "legal interests."
            )

        if (
            "collection" in combined
            or "service charge" in combined
            or "delinquent" in combined
        ):
            return (
                "Client shall be responsible only for reasonable and "
                "documented collection costs and any lawful service "
                "charge expressly stated in this Agreement, after "
                "Law Firm provides written notice of the overdue "
                "amount and a reasonable opportunity to cure. Any "
                "disputed amount shall be handled through a "
                "good-faith billing dispute process before collection "
                "costs or service charges are imposed."
            )

        if (
            "privilege" in combined
            or "confidentiality waiver" in combined
            or "confidential" in combined
        ):
            return (
                "In any joint representation, Law Firm shall explain "
                "in writing the confidentiality and privilege "
                "consequences of joint representation, and Client's "
                "consent shall be informed and written. Confidential "
                "information shall be protected except to the extent "
                "disclosure between jointly represented clients is "
                "necessary for the representation or required by "
                "applicable professional rules or law."
            )

        if "rate change" in combined or "fee increase" in combined or "fees" in combined:
            return (
                "Any change to hourly rates or charges shall apply "
                "only prospectively after reasonable prior written "
                "notice to Client. Client may decline the changed "
                "rates for future work, and Law Firm shall cooperate "
                "reasonably in transitioning the matter in accordance "
                "with applicable professional conduct obligations."
            )

    # --------------------------------------------------------
    # EMPLOYMENT
    # --------------------------------------------------------

    if context == "employment":

        if "termination" in combined or "notice" in combined:
            return (
                "The Employer may terminate the Employee's employment "
                "only by providing reasonable prior written notice "
                "specifying the grounds for termination and allowing "
                "the Employee a reasonable opportunity to cure any "
                "remediable breach. Immediate termination shall be "
                "permitted only where legally justified, including "
                "proven serious misconduct, fraud, or wilful breach, "
                "subject to applicable law and the Employee's accrued "
                "lawful entitlements."
            )

    # --------------------------------------------------------
    # LEASE
    # --------------------------------------------------------

    if context == "lease":

        if (
            "termination" in combined
            or "notice" in combined
            or "suspension" in combined
        ):
            return (
                "Either the Landlord or Tenant may terminate this "
                "Agreement only after reasonable prior written notice "
                "stating the applicable grounds and, where reasonably "
                "possible, providing a reasonable opportunity to cure "
                "any remediable breach, subject to the rights and "
                "remedies available under applicable law."
            )

    # --------------------------------------------------------
    # LOAN
    # --------------------------------------------------------

    if context == "loan":

        if (
            "termination" in combined
            or "default" in combined
            or "acceleration" in combined
        ):
            return (
                "Any enforcement action following a default shall "
                "require written notice specifying the default and, "
                "where permitted by applicable law and this Agreement, "
                "a reasonable opportunity to cure before acceleration "
                "or other material enforcement action is taken."
            )

    # --------------------------------------------------------
    # SERVICE AGREEMENT
    # --------------------------------------------------------

    if context == "service_agreement":

        if (
            "termination" in combined
            or "suspension" in combined
            or "notice" in combined
        ):
            return (
                "Either party may terminate or suspend the services "
                "only after reasonable prior written notice specifying "
                "the applicable grounds and, where reasonably possible, "
                "a reasonable opportunity to cure any remediable breach, "
                "subject to applicable law and the terms of this Agreement."
            )

    # --------------------------------------------------------
    # GENERAL TERMINATION
    # --------------------------------------------------------

    if "termination" in combined or "suspension" in combined or "notice" in combined:
        return (
            "Either party may terminate or suspend performance only "
            "after providing reasonable prior written notice specifying "
            "the grounds for such action and allowing a reasonable "
            "opportunity to cure any remediable breach, except where "
            "immediate action is required by law, fraud, wilful "
            "misconduct, or urgent risk of material harm."
        )

    # --------------------------------------------------------
    # LIEN
    # --------------------------------------------------------

    if "lien" in combined or "payment security" in combined:
        return (
            "Any lien or payment security shall be limited to "
            "undisputed amounts that are lawfully due and reasonable "
            "documented costs, and shall remain subject to applicable "
            "law, a good-faith dispute process, and the rights of "
            "the other party to access property or documents "
            "necessary to protect its legal interests."
        )

    # --------------------------------------------------------
    # COLLECTION
    # --------------------------------------------------------

    if "collection" in combined or "service charge" in combined or "delinquent" in combined:
        return (
            "The defaulting party shall be responsible only for "
            "reasonable documented collection costs and any lawful "
            "service charge expressly stated in this Agreement, "
            "after receiving written notice of the overdue amount "
            "and a reasonable opportunity to cure or dispute the "
            "amount in good faith."
        )

    # --------------------------------------------------------
    # PRIVILEGE
    # --------------------------------------------------------

    if "privilege" in combined or "confidentiality waiver" in combined:
        return (
            "Confidential information and privileged communications "
            "shall remain protected except to the extent disclosure "
            "is required by applicable law, professional obligations, "
            "or informed written consent of the affected parties "
            "after clear disclosure of the consequences of such waiver."
        )

    # --------------------------------------------------------
    # LIABILITY
    # --------------------------------------------------------

    if "liability" in combined:
        return (
            "Each party shall be liable only for direct losses caused "
            "by its proven breach, negligence, fraud, or wilful misconduct, "
            "and such liability shall be subject to a reasonable and "
            "mutually agreed cap, except where limitation is prohibited "
            "by applicable law."
        )

    # --------------------------------------------------------
    # INDEMNITY
    # --------------------------------------------------------

    if "indemn" in combined:
        return (
            "Each party shall indemnify the other only for direct "
            "losses arising from its own breach, negligence, fraud, "
            "or wilful misconduct, subject to a reasonable liability "
            "cap and applicable law."
        )

    # --------------------------------------------------------
    # AMENDMENT / MODIFICATION
    # --------------------------------------------------------

    if (
        "amend" in combined
        or "modify" in combined
        or "rate change" in combined
        or "fee increase" in combined
    ):
        return (
            "Any amendment, modification, fee increase, or material "
            "change to this Agreement shall apply only after reasonable "
            "prior written notice and, where it affects material rights "
            "or obligations, written agreement of both parties before "
            "taking effect."
        )

    # --------------------------------------------------------
    # CONFIDENTIALITY
    # --------------------------------------------------------

    if "confidential" in combined:
        return (
            "The receiving party shall protect confidential information "
            "with reasonable care and shall not disclose it except as "
            "required by law, for legitimate contractual purposes, or "
            "with prior written consent. This obligation shall continue "
            "for a reasonable period after termination, except for trade "
            "secrets which shall remain protected for as long as legally valid."
        )

    # --------------------------------------------------------
    # GENERIC FALLBACK
    # --------------------------------------------------------

    return (
        "The parties shall act reasonably and in good faith, and any "
        "material action affecting rights, obligations, liability, "
        "termination, or payment shall require prior written notice, "
        "a reasonable opportunity to cure where applicable, and "
        "compliance with applicable law."
    )


# ============================================================
# NORMALIZE LLM RESPONSE
# ============================================================

def normalize_reasoning(result: dict, risk_result: dict, clause: str = "") -> dict:
    """
    Ensure all required keys exist.

    Prevents incorrect employment-specific safer clauses.
    """

    if not isinstance(result, dict):
        result = {}

    # --------------------------------------------------------
    # Reason
    # --------------------------------------------------------

    reason = result.get("reason")

    if not reason:
        reason = (
            f"The classifier identified this clause as "
            f"{risk_result.get('risk_level')} risk because of "
            f"{risk_result.get('detected_issue')}."
        )

    # --------------------------------------------------------
    # Action
    # --------------------------------------------------------

    action = result.get("action")

    if not action:
        action = (
            "The signer should review and negotiate "
            "this clause before signing."
        )

    # --------------------------------------------------------
    # Safer Clause
    # --------------------------------------------------------

    safer_clause = result.get("safer_clause")

    if not safer_clause:
        safer_clause = get_default_safer_clause(risk_result, clause)

    # --------------------------------------------------------
    # Context validation
    # --------------------------------------------------------

    context = infer_clause_context(clause, risk_result)

    if safer_clause_context_mismatch(clause, safer_clause, context):
        safer_clause = get_default_safer_clause(risk_result, clause)

    # --------------------------------------------------------
    # Negotiation
    #
    # Kept only for compatibility.
    # analyzer_v2.py should use negotiation_engine.py.
    # --------------------------------------------------------

    negotiation_advice = result.get("negotiation_advice", "")

    if not negotiation_advice:
        negotiation_advice = (
            "The signer should request clearer, fairer, "
            "and mutually balanced wording before signing."
        )

    return {
        "reason": clean_generated_text(reason),
        "action": clean_generated_text(action),
        "safer_clause": clean_generated_text(safer_clause),
        "negotiation_advice": clean_generated_text(negotiation_advice),
    }


# ============================================================
# QWEN / STATIC FALLBACK
# ============================================================

QWEN_SUB_BATCH_SIZE = 3


def _build_batch_reasoning_prompt(items):
    blocks = []
    for idx, item in enumerate(items, start=1):
        single = build_reasoning_prompt(
            clause=item["clause"],
            risk_result=item["risk_result"],
            similar_clauses=item.get("similar_clauses"),
        )
        blocks.append(f"\n### CLAUSE_ID: {idx}\n{single}\n")

    joined = "\n".join(blocks)

    return f"""
You will receive MULTIPLE clause reasoning tasks below,
each marked with CLAUSE_ID.

Return ONLY valid JSON in this exact format, one entry
per CLAUSE_ID, in the same order:

{{
  "results": [
    {{
      "clause_id": 1,
      "reason": "...",
      "action": "...",
      "safer_clause": "..."
    }}
  ]
}}

{joined}
"""


def _parse_batch_results(raw, items):
    data = extract_json(raw)

    if isinstance(data, dict):
        results = data.get("results", [])
    elif isinstance(data, list):
        results = data
    else:
        raise ValueError("Unexpected batch reasoning JSON structure")

    if not isinstance(results, list) or len(results) != len(items):
        raise ValueError(
            f"Batch reasoning returned {len(results) if isinstance(results, list) else 'invalid'} "
            f"results for {len(items)} clauses"
        )

    normalized = []
    for item, result in zip(items, results):
        normalized.append(
            normalize_reasoning(result, item["risk_result"], item["clause"])
        )
    return normalized

def is_transient_error(e) -> bool:
    """503/500/timeouts are temporary. 429 (quota) and 404 are NOT."""
    status = getattr(e, "status_code", None)
    text = str(e)
    return (
        status in (500, 502, 503, 504)
        or "UNAVAILABLE" in text
        or "timed out" in text.lower()
    )


def call_with_transient_retry(fn, attempts=3, delay=5):
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as e:
            if attempt < attempts and is_transient_error(e):
                wait = delay * attempt
                print(f"[GEMINI] Temporary error (attempt {attempt}/{attempts}), retrying in {wait}s...")
                time.sleep(wait)
                continue
            raise

def is_gemini_available() -> bool:
    return _gemini_available


def mark_gemini_unavailable():
    global _gemini_available
    _gemini_available = False


def _reasoning_cache_key(item: dict) -> str:
    return llm_cache.make_key(
        "reason", MODEL_NAME, item["clause"],
        item["risk_result"].get("detected_issue", "")
    )


def _generate_reasoning_batch_uncached(items: list):
    if is_gemini_available():
        prompt = _build_batch_reasoning_prompt(items)
        try:
            print(f"[LLM] Calling model: {MODEL_NAME} (batch reasoning, {len(items)} clauses)")
            response = call_with_transient_retry(lambda: client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": (
                        "You are a professional legal contract review assistant. "
                        "Return ONLY valid JSON with a 'results' array.")},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800 * len(items)
            ))
            raw = response.choices[0].message.content
            return _parse_batch_results(raw, items), True
        except Exception as e:
            print(f"[LLM ERROR] Batch reasoning failed: {e}")
            print("[LLM] Gemini skipped for the rest of this document.")
            mark_gemini_unavailable()

    print(f"[LLM] Using local Qwen fallback (sub-batches of {QWEN_SUB_BATCH_SIZE})...")
    results = []
    for start in range(0, len(items), QWEN_SUB_BATCH_SIZE):
        sub_items = items[start:start + QWEN_SUB_BATCH_SIZE]
        print(f"[QWEN] Reasoning sub-batch {start + 1}-{start + len(sub_items)} of {len(items)}...")
        try:
            raw_qwen = local_llm.generate(
                system_prompt=(
                    "You are a professional legal contract review assistant. "
                    "Return ONLY valid JSON with a 'results' array."),
                user_prompt=_build_batch_reasoning_prompt(sub_items),
                max_new_tokens=800 * len(sub_items),
            )
            results.extend(_parse_batch_results(raw_qwen, sub_items))
        except Exception as e:
            print(f"[QWEN] Reasoning sub-batch failed: {e} — using per-clause fallback")
            for item in sub_items:
                results.append(
                    _reason_with_qwen_or_fallback(
                        build_reasoning_prompt(item["clause"], item["risk_result"], item.get("similar_clauses")),
                        item["risk_result"], item["clause"]))
    return results, False


def generate_reasoning_batch(items: list) -> list:
    if not items:
        return []

    results = [None] * len(items)
    pending = []

    for i, item in enumerate(items):
        cached = llm_cache.get(_reasoning_cache_key(item))
        if cached:
            results[i] = cached
        else:
            pending.append(i)

    if len(pending) < len(items):
        print(f"[CACHE] {len(items) - len(pending)}/{len(items)} reasoning results served from cache (no Gemini call)")

    if pending:
        pending_items = [items[i] for i in pending]
        fresh, used_gemini = _generate_reasoning_batch_uncached(pending_items)
        if used_gemini:
            llm_cache.put_many([
                (_reasoning_cache_key(it), res) for it, res in zip(pending_items, fresh)
            ])
        for i, res in zip(pending, fresh):
            results[i] = res

    return results
def _reason_with_qwen_or_fallback(prompt, risk_result, clause):

    try:
        raw_qwen = local_llm.generate(
            system_prompt=(
                "You are a professional legal contract "
                "review assistant. Return ONLY valid JSON "
                "with exactly three fields: reason, action, "
                "and safer_clause."
            ),
            user_prompt=prompt,
            max_new_tokens=800,
        )
        qwen_result = extract_json(raw_qwen)
        if not qwen_result:
            raise ValueError("Qwen response could not be parsed as JSON")
        missing = [f for f in ("reason", "action", "safer_clause") if not qwen_result.get(f)]
        if missing:
            raise ValueError("Qwen JSON missing required fields: " + ", ".join(missing))
        print("[LOCAL LLM] Qwen reasoning succeeded.")
        return normalize_reasoning(qwen_result, risk_result, clause)
    except Exception as qwen_error:
        print(f"[LOCAL LLM] Qwen reasoning failed: {qwen_error}")

    print("[LLM] Using static local fallback.")
    fallback = {
        "reason": (
            f"The classifier identified this clause as "
            f"{risk_result.get('risk_level')} risk because "
            f"of {risk_result.get('detected_issue')}."
        ),
        "action": "The signer should review and negotiate this clause before signing.",
        "safer_clause": get_default_safer_clause(risk_result, clause),
        "negotiation_advice": (
            "The signer should request clearer, fairer, "
            "and mutually balanced wording before signing."
        ),
    }
    return normalize_reasoning(fallback, risk_result, clause)


# ============================================================
# GENERATE REASONING
# ============================================================

def generate_reasoning(clause: str, risk_result: dict, similar_clauses=None) -> dict:
    """
    Generate:
    - legal reason
    - recommended action
    - safer clause

    Risk classification is NOT performed by the LLM.
    """

    global _gemini_available

    prompt = build_reasoning_prompt(
        clause=clause,
        risk_result=risk_result,
        similar_clauses=similar_clauses,
    )

    if not _gemini_available:
        print("[LLM] Gemini marked unavailable for this document. Using local Qwen directly.")
        return _reason_with_qwen_or_fallback(prompt, risk_result, clause)

    try:
        print(f"[LLM] Calling model: {MODEL_NAME}")
        print(f"[LLM] Base URL: {settings.LLM_BASE_URL}")

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a professional legal contract "
                        "review assistant. Return ONLY valid JSON "
                        "with exactly three fields: reason, action, "
                        "and safer_clause."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=800,
        )

        # ----------------------------------------------------
        # Validate response
        # ----------------------------------------------------

        if not response:
            raise ValueError("LLM returned an empty response object")

        if not response.choices:
            raise ValueError("LLM returned no choices")

        message = response.choices[0].message

        # ----------------------------------------------------
        # Get normal content
        # ----------------------------------------------------

        raw = getattr(message, "content", None)

        # ----------------------------------------------------
        # Some reasoning models/providers may return
        # content differently. Try string conversion.
        # ----------------------------------------------------

        if not raw:
            raw = str(message)

        if not raw or not str(raw).strip():
            raise ValueError("LLM returned empty message content")

        raw = str(raw).strip()

        print("[LLM] Response received successfully.")
        print("[LLM] Raw response preview:")
        print(raw[:500])

        # ----------------------------------------------------
        # Parse JSON
        # ----------------------------------------------------

        result = extract_json(raw)

        if result is None:
            print("[LLM] JSON parsing failed.")
            raise ValueError("LLM response could not be parsed as JSON")

        # ----------------------------------------------------
        # Validate required fields
        # ----------------------------------------------------

        required_fields = ["reason", "action", "safer_clause"]

        missing_fields = [field for field in required_fields if not result.get(field)]

        if missing_fields:
            raise ValueError(
                "LLM JSON missing required fields: " + ", ".join(missing_fields)
            )

        # ----------------------------------------------------
        # Normalize
        # ----------------------------------------------------

        return normalize_reasoning(result, risk_result, clause)

    except Exception as e:

        # ----------------------------------------------------
        # Print actual error
        # ----------------------------------------------------

        print("\n[LLM ERROR]")
        print(f"Model: {MODEL_NAME}")
        print(f"Base URL: {settings.LLM_BASE_URL}")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error: {str(e)}")
        print("[LLM] Gemini failed. Trying local Qwen fallback.")

        _gemini_available = False

        return _reason_with_qwen_or_fallback(prompt, risk_result, clause)