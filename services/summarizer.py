"""
LexiShield — Professional Legal Document Summarizer

Summarization:
    - BART local transformer model
    - Token-aware chunking
    - Overlapping chunks for long legal text
    - Hierarchical summarization for long documents
    - Rule-based legal information extraction
"""

import re
import torch
from collections import OrderedDict

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)

from services.config import settings


# ============================================================
# BART CONFIGURATION
# ============================================================

BART_MODEL_NAME = settings.SUMMARY_MODEL_NAME

BART_INPUT_TOKENS = settings.SUMMARY_MAX_INPUT_TOKENS

BART_MIN_LENGTH = settings.SUMMARY_MIN_LENGTH

BART_MAX_LENGTH = settings.SUMMARY_MAX_LENGTH

FINAL_SUMMARY_MAX_INPUT_TOKENS = (
    settings.SUMMARY_MAX_INPUT_TOKENS
)

# Overlap helps preserve context when a long
# legal clause is split between two chunks.
BART_CHUNK_OVERLAP = 80


# ============================================================
# LOAD BART LAZILY
# ============================================================

tokenizer = None
model = None


def load_bart_model():
    """
    Load the local BART model only when summarization
    is actually requested.
    """

    global tokenizer
    global model

    if tokenizer is not None and model is not None:
        return

    print(
        f"[Summarizer] Loading BART model: "
        f"{BART_MODEL_NAME}"
    )

    try:

        tokenizer = AutoTokenizer.from_pretrained(
            BART_MODEL_NAME
        )

        model = AutoModelForSeq2SeqLM.from_pretrained(
            BART_MODEL_NAME
        )

        model.eval()

        print(
            "[Summarizer] BART loaded successfully ✓"
        )

    except Exception as e:

        tokenizer = None
        model = None

        print(
            "[Summarizer] BART loading failed:"
            f" {e}"
        )

        raise RuntimeError(
            f"Unable to load BART model: {e}"
        )


# ============================================================
# DOCUMENT TYPES
# ============================================================

DOCUMENT_TYPES = {

    "Employment Contract": [
        "employment agreement",
        "employment contract",
        "employee",
        "employer",
        "designation",
        "probation",
        "salary",
        "ctc",
        "duties and powers",
        "termination of employment",
        "non-compete",
        "severance pay",
    ],

    "Rental / Lease Agreement": [
        "tenant",
        "landlord",
        "lease agreement",
        "rent",
        "security deposit",
        "premises",
        "vacate",
        "possession",
    ],

    "Sale Agreement": [
        "buyer",
        "seller",
        "sale agreement",
        "purchase",
        "sale price",
        "transfer of ownership",
    ],

    "Loan Agreement": [
        "borrower",
        "lender",
        "loan",
        "interest rate",
        "emi",
        "repayment",
        "principal amount",
    ],

    "Non-Disclosure Agreement": [
        "non-disclosure",
        "confidentiality agreement",
        "trade secret",
        "confidential information",
    ],

    "Service Agreement": [
        "service provider",
        "scope of work",
        "deliverables",
        "milestone",
        "service charges",
        "invoice",
    ],
}


# ============================================================
# SECTION MAP
# ============================================================

SECTION_MAP = {

    "1": "Designation",
    "2": "Probationary Period",
    "3": "Place of Work",
    "4": "Duties and Powers",
    "5": "Annual Base Salary",
    "6": "Confidentiality",
    "7": "Non-Compete Clause",
    "8": "Term, Termination and Severance Pay",
    "9": "Other Terms of Employment",
    "10": "Miscellaneous",
    "11": "Jurisdiction",
}


# ============================================================
# BASIC CLEANING
# ============================================================

def clean_text(text):

    if not text:
        return ""

    text = str(text)

    text = text.replace(
        "\r",
        "\n"
    )

    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    text = re.sub(
        r" *\n *",
        "\n",
        text
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    text = re.sub(
        r" +([,.;:!?])",
        r"\1",
        text
    )

    return text.strip()


# ============================================================
# DEDUPLICATION
# ============================================================

def dedupe(
    items,
    limit=8
):

    result = []
    seen = set()

    for item in items:

        item = clean_text(
            item
        )

        if not item:
            continue

        key = item.lower()

        if key in seen:
            continue

        seen.add(key)

        result.append(
            item
        )

        if len(result) >= limit:
            break

    return result


# ============================================================
# DOCUMENT TYPE
# ============================================================

def detect_document_type(text):

    lower_text = text.lower()

    scores = OrderedDict()

    for (
        document_type,
        keywords
    ) in DOCUMENT_TYPES.items():

        score = 0

        for keyword in keywords:

            if keyword in lower_text:

                if len(
                    keyword.split()
                ) > 1:

                    score += 3

                else:

                    score += 2

        scores[
            document_type
        ] = score

    best_type = max(
        scores,
        key=scores.get
    )

    if scores[
        best_type
    ] < 3:

        return "Legal Document"

    return best_type


# ============================================================
# PARTIES
# ============================================================

def extract_parties(text):

    parties = []

    if re.search(
        r"\bemployer\b",
        text,
        re.IGNORECASE
    ):

        parties.append(
            "Employer / Company"
        )

    if re.search(
        r"\bemployee\b",
        text,
        re.IGNORECASE
    ):

        parties.append(
            "Employee"
        )

    if re.search(
        r"\blandlord\b",
        text,
        re.IGNORECASE
    ):

        parties.append(
            "Landlord"
        )

    if re.search(
        r"\btenant\b",
        text,
        re.IGNORECASE
    ):

        parties.append(
            "Tenant"
        )

    if re.search(
        r"\bbuyer\b",
        text,
        re.IGNORECASE
    ):

        parties.append(
            "Buyer"
        )

    if re.search(
        r"\bseller\b",
        text,
        re.IGNORECASE
    ):

        parties.append(
            "Seller"
        )

    if re.search(
        r"\bborrower\b",
        text,
        re.IGNORECASE
    ):

        parties.append(
            "Borrower"
        )

    if re.search(
        r"\blender\b",
        text,
        re.IGNORECASE
    ):

        parties.append(
            "Lender"
        )

    return dedupe(
        parties,
        8
    )


# ============================================================
# FINANCIAL TERMS
# ============================================================

def extract_financial_terms(text):
    """
    Extract and normalize financial terms.

    Handles OCR line breaks such as:

        Rs. 2
        lakh
    """

    normalized_text = re.sub(
        r"(?i)(rs\.?|inr|₹)\s*\n\s*",
        r"\1 ",
        text
    )

    normalized_text = re.sub(
        r"(?i)(\d[\d,]*(?:\.\d+)?)\s*\n\s*"
        r"(lakh|lakhs|crore|crores|rupees)\b",
        r"\1 \2",
        normalized_text
    )

    text_lower = normalized_text.lower()

    financial_terms = []

    patterns = [

        r"(?:rs\.?|inr|₹)\s*"
        r"[\d,]+(?:\.\d+)?\s*"
        r"(?:lakh|lakhs|crore|crores)?",

        r"\b[\d,]+(?:\.\d+)?\s*"
        r"(?:lakh|lakhs|crore|crores|rupees)\b"
    ]

    for pattern in patterns:

        matches = re.findall(
            pattern,
            normalized_text,
            re.IGNORECASE
        )

        for amount in matches:

            amount = clean_text(
                amount
            )

            normalized = (
                amount.lower()
            )

            normalized = normalized.replace(
                "rs.",
                "rs"
            )

            normalized = normalized.replace(
                "lakhs",
                "lakh"
            )

            normalized = normalized.replace(
                "crores",
                "crore"
            )

            normalized = re.sub(
                r"\s+",
                " ",
                normalized
            ).strip()

            existing = []

            for item in financial_terms:

                item_normalized = (
                    item.lower()
                )

                item_normalized = (
                    item_normalized.replace(
                        "rs.",
                        "rs"
                    )
                )

                item_normalized = (
                    item_normalized.replace(
                        "lakhs",
                        "lakh"
                    )
                )

                item_normalized = (
                    item_normalized.replace(
                        "crores",
                        "crore"
                    )
                )

                item_normalized = re.sub(
                    r"\s+",
                    " ",
                    item_normalized
                ).strip()

                existing.append(
                    item_normalized
                )

            if normalized not in existing:

                financial_terms.append(
                    re.sub(
                        r"\s+",
                        " ",
                        amount
                    ).strip()
                )

    if re.search(
        r"salary.*(?:ctc|payroll)|ctc.*salary",
        text_lower
    ):

        financial_terms.append(
            "CTC / salary terms"
        )

    if re.search(
        r"2\s*(?:year|years).*bond|"
        r"bond.*2\s*(?:year|years)",
        text_lower
    ):

        financial_terms.append(
            "Two-year employment bond"
        )

    if re.search(
        r"training.*6\s*months|"
        r"6\s*months.*training",
        text_lower
    ):

        financial_terms.append(
            "Training-cost recovery before 6 months"
        )

    return dedupe(
        financial_terms,
        8
    )


# ============================================================
# IMPORTANT CLAUSES
# ============================================================

def extract_important_clauses(text):

    clauses = []

    matches = re.finditer(
        r"(?m)^\s*(\d{1,2})\.\s*([A-Z][^\n]{2,100})",
        text
    )

    for match in matches:

        number = match.group(1)

        title = clean_text(
            match.group(2)
        )

        if number in SECTION_MAP:

            clauses.append(
                f"Section {number} — "
                f"{SECTION_MAP[number]}"
            )

        elif title:

            clauses.append(
                f"Section {number} — "
                f"{title}"
            )

    lower_text = text.lower()

    important_keywords = [

        (
            "Arbitration / Dispute Resolution",
            "arbitration"
        ),

        (
            "Confidentiality",
            "confidential"
        ),

        (
            "Intellectual Property",
            "source code"
        ),

        (
            "Non-Compete",
            "non-compete"
        ),

        (
            "Termination",
            "termination"
        ),

        (
            "Jurisdiction",
            "jurisdiction"
        ),
    ]

    for (
        title,
        keyword
    ) in important_keywords:

        if keyword in lower_text:

            clauses.append(
                title
            )

    return dedupe(
        clauses,
        10
    )


# ============================================================
# KEY POINTS
# ============================================================

def extract_key_points(
    text,
    document_type
):

    lower_text = text.lower()

    points = []

    if document_type == "Employment Contract":

        rules = [

            (
                "Probationary period and performance expectations",
                ["probation"]
            ),

            (
                "Work location and transfer conditions",
                [
                    "place of work",
                    "transfer"
                ]
            ),

            (
                "Employee duties and responsibilities",
                ["duties and powers"]
            ),

            (
                "Salary / CTC terms",
                [
                    "salary",
                    "ctc"
                ]
            ),

            (
                "Confidentiality and company information",
                ["confidentiality"]
            ),

            (
                "Company ownership of work product / source code",
                ["source code"]
            ),

            (
                "Post-employment non-compete restriction",
                ["non-compete"]
            ),

            (
                "Termination and notice conditions",
                [
                    "termination",
                    "notice"
                ]
            ),

            (
                "Early-exit penalty / bond condition",
                [
                    "2 lakh",
                    "2 lakhs",
                    "two years"
                ]
            ),

            (
                "Training-cost recovery condition",
                [
                    "training",
                    "6 months"
                ]
            ),

            (
                "Arbitration and jurisdiction",
                [
                    "arbitration",
                    "jurisdiction"
                ]
            ),
        ]

        for (
            title,
            keywords
        ) in rules:

            if any(
                keyword in lower_text
                for keyword in keywords
            ):

                points.append(
                    title
                )

    else:

        important_words = [

            "payment",
            "termination",
            "notice",
            "confidentiality",
            "liability",
            "penalty",
            "arbitration",
            "jurisdiction"
        ]

        for word in important_words:

            if word in lower_text:

                points.append(
                    word.capitalize()
                    + " terms are included."
                )

    return dedupe(
        points,
        12
    )


# ============================================================
# LEGAL TERMS
# ============================================================

def extract_legal_terms(text):

    legal_terms = [

        "agreement",
        "employment",
        "probation",
        "designation",
        "confidentiality",
        "trade secret",
        "intellectual property",
        "source code",
        "non-compete",
        "termination",
        "severance",
        "penalty",
        "bond",
        "arbitration",
        "arbitrator",
        "jurisdiction",
        "liability",
        "breach",
        "damages",
        "dispute",
        "claim",
        "notice period",
        "misconduct",
        "insubordination",
        "indemnity",
        "obligation",
        "contract",
        "compensation",
        "legal proceeding"
    ]

    lower_text = text.lower()

    found = []

    for term in legal_terms:

        if term in lower_text:

            found.append(
                term.title()
            )

    return dedupe(
        found,
        20
    )


# ============================================================
# KEY INFORMATION
# ============================================================

def extract_key_info(text):

    dates = []

    date_patterns = [

        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",

        r"\b\d{1,2}\s+"
        r"(?:January|February|March|April|May|June|July|August|"
        r"September|October|November|December)\s+\d{4}\b",

        r"\b(?:January|February|March|April|May|June|July|August|"
        r"September|October|November|December)\s+\d{1,2},\s+\d{4}\b"
    ]

    for pattern in date_patterns:

        dates.extend(
            re.findall(
                pattern,
                text,
                re.IGNORECASE
            )
        )

    document_type = detect_document_type(
        text
    )

    return {

        "parties": extract_parties(
            text
        ),

        "dates": dedupe(
            dates,
            8
        ),

        "amounts": extract_financial_terms(
            text
        ),

        "clauses": extract_important_clauses(
            text
        ),

        "key_points": extract_key_points(
            text,
            document_type
        ),

        "legal_terms": extract_legal_terms(
            text
        )
    }


# ============================================================
# BART CHUNKING
# ============================================================

def create_bart_chunks(
    text,
    max_tokens=BART_INPUT_TOKENS
):
    """
    Split long legal text into token-safe chunks.

    Paragraph boundaries are preferred.

    Long paragraphs are split by tokens with a small
    overlap so important legal context is not lost.
    """

    if tokenizer is None:
        return []

    text = clean_text(
        text
    )

    if not text:
        return []

    paragraphs = re.split(
        r"\n{2,}",
        text
    )

    paragraphs = [
        clean_text(p)
        for p in paragraphs
        if clean_text(p)
    ]

    chunks = []
    current_parts = []
    current_tokens = 0

    for paragraph in paragraphs:

        paragraph_tokens = tokenizer.encode(
            paragraph,
            add_special_tokens=False
        )

        paragraph_token_count = len(
            paragraph_tokens
        )

        # ----------------------------------------------------
        # Very long paragraph
        # ----------------------------------------------------

        if paragraph_token_count > max_tokens:

            if current_parts:

                chunks.append(
                    "\n\n".join(
                        current_parts
                    )
                )

                current_parts = []
                current_tokens = 0

            start = 0

            while start < paragraph_token_count:

                end = min(
                    start + max_tokens,
                    paragraph_token_count
                )

                token_slice = (
                    paragraph_tokens[
                        start:end
                    ]
                )

                chunk_text = tokenizer.decode(
                    token_slice,
                    skip_special_tokens=True
                )

                chunk_text = clean_text(
                    chunk_text
                )

                if chunk_text:

                    chunks.append(
                        chunk_text
                    )

                # Preserve context between chunks.
                next_start = (
                    end - BART_CHUNK_OVERLAP
                )

                # Safety guard against infinite loops.
                if next_start <= start:

                    next_start = end

                start = next_start

            continue

        # ----------------------------------------------------
        # Add paragraph to current chunk
        # ----------------------------------------------------

        if (
            current_tokens
            + paragraph_token_count
            <= max_tokens
        ):

            current_parts.append(
                paragraph
            )

            current_tokens += (
                paragraph_token_count
            )

        else:

            if current_parts:

                chunks.append(
                    "\n\n".join(
                        current_parts
                    )
                )

            current_parts = [
                paragraph
            ]

            current_tokens = (
                paragraph_token_count
            )

    # --------------------------------------------------------
    # Remaining chunk
    # --------------------------------------------------------

    if current_parts:

        chunks.append(
            "\n\n".join(
                current_parts
            )
        )

    return chunks


# ============================================================
# BART SINGLE CHUNK
# ============================================================

def summarize_chunk(
    text,
    min_length=BART_MIN_LENGTH,
    max_length=BART_MAX_LENGTH
):
    """
    Summarize one token-safe chunk using local BART.
    """

    if tokenizer is None or model is None:

        raise RuntimeError(
            "BART model is not available."
        )

    text = clean_text(
        text
    )

    if not text:
        return ""

    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=BART_INPUT_TOKENS,
        truncation=True
    )

    input_length = (
        inputs["input_ids"].shape[1]
    )

    # Very short input does not need
    # forced abstractive summarization.
    if input_length < 40:

        return text

    # Keep generation parameters valid
    # for the actual input length.
    safe_max_length = max(
        30,
        min(
            max_length,
            input_length - 5
        )
    )

    safe_min_length = min(
        min_length,
        max(
            15,
            safe_max_length - 10
        )
    )

    if safe_min_length >= safe_max_length:

        safe_min_length = max(
            10,
            safe_max_length // 2
        )

    with torch.inference_mode():

        summary_ids = model.generate(

            **inputs,

            num_beams=4,

            length_penalty=2.0,

            max_length=safe_max_length,

            min_length=safe_min_length,

            no_repeat_ngram_size=3,

            early_stopping=True
        )

    summary = tokenizer.decode(
        summary_ids[0],
        skip_special_tokens=True
    )

    return clean_text(
        summary
    )


# ============================================================
# HIERARCHICAL BART SUMMARY
# ============================================================

def generate_bart_summary(
    text
):
    """
    Generate a meaningful summary for a legal document.

    Short document:
        Text → BART

    Long document:
        Text → chunks → BART per chunk
             → combined summaries
             → BART final summary
    """

    # Load only when summarization is requested.
    load_bart_model()

    text = clean_text(
        text
    )

    if not text:

        return ""

    chunks = create_bart_chunks(
        text
    )

    print(
        f"[Summarizer] BART chunks: "
        f"{len(chunks)}"
    )

    if not chunks:

        return ""

    # --------------------------------------------------------
    # First pass
    # --------------------------------------------------------

    chunk_summaries = []

    for index, chunk in enumerate(
        chunks,
        start=1
    ):

        print(
            f"[Summarizer] "
            f"BART chunk {index}/{len(chunks)}"
        )

        try:

            summary = summarize_chunk(
                chunk
            )

        except Exception as e:

            print(
                f"[Summarizer] "
                f"Chunk {index} failed: {e}"
            )

            summary = ""

        if summary:

            chunk_summaries.append(
                summary
            )

    if not chunk_summaries:

        return ""

    combined_summary = "\n\n".join(
        chunk_summaries
    )

    # --------------------------------------------------------
    # Final pass for multiple chunks
    # --------------------------------------------------------

    combined_tokens = tokenizer.encode(
        combined_summary,
        add_special_tokens=False
    )

    if (
        len(combined_tokens)
        <= FINAL_SUMMARY_MAX_INPUT_TOKENS
    ):

        final_summary = summarize_chunk(
            combined_summary,
            min_length=BART_MIN_LENGTH,
            max_length=BART_MAX_LENGTH
        )

        return clean_text(
            final_summary
        )

    # --------------------------------------------------------
    # If combined summaries are still too long,
    # summarize them in groups.
    # --------------------------------------------------------

    final_chunks = create_bart_chunks(
        combined_summary,
        max_tokens=FINAL_SUMMARY_MAX_INPUT_TOKENS
    )

    second_pass = []

    for index, chunk in enumerate(
        final_chunks,
        start=1
    ):

        print(
            f"[Summarizer] "
            f"Final BART pass "
            f"{index}/{len(final_chunks)}"
        )

        try:

            summary = summarize_chunk(
                chunk,
                min_length=BART_MIN_LENGTH,
                max_length=BART_MAX_LENGTH
            )

        except Exception as e:

            print(
                f"[Summarizer] "
                f"Final pass failed: {e}"
            )

            summary = ""

        if summary:

            second_pass.append(
                summary
            )

    if not second_pass:

        return clean_text(
            combined_summary
        )

    final_text = "\n\n".join(
        second_pass
    )

    # --------------------------------------------------------
    # One last compression pass only if safely
    # within BART input limits.
    # --------------------------------------------------------

    final_tokens = tokenizer.encode(
        final_text,
        add_special_tokens=False
    )

    if (
        len(final_tokens)
        <= FINAL_SUMMARY_MAX_INPUT_TOKENS
    ):

        try:

            final_text = summarize_chunk(
                final_text,
                min_length=BART_MIN_LENGTH,
                max_length=BART_MAX_LENGTH
            )

        except Exception as e:

            print(
                f"[Summarizer] "
                f"Final compression failed: {e}"
            )

    return clean_text(
        final_text
    )


# ============================================================
# STRUCTURED SUMMARY (professional JSON output)
# ============================================================

def build_structured_summary(
    bart_summary,
    key_info,
    doc_type,
    stats
):
    """
    Return the summary as a structured dict instead of a
    single flat text blob, so the frontend can render each
    section (parties, dates, amounts, clauses, etc.) as its
    own UI element instead of parsing plain text.
    """

    return {
        "document_type": doc_type,

        "ai_summary": (
            bart_summary
            if bart_summary
            else "A summary could not be generated."
        ),

        "parties": key_info.get("parties", []),

        "important_dates": key_info.get("dates", []),

        "financial_terms": key_info.get("amounts", []),

        "key_terms_and_obligations": [
            point.strip()
            for point in key_info.get("key_points", [])
            if point and point.strip()
        ],

        "important_clauses": key_info.get("clauses", []),

        "key_legal_terms": key_info.get("legal_terms", []),

        "statistics": {
            "total_words": stats.get("total_words", 0),
            "total_sentences": stats.get("total_sentences", 0),
            "legal_terms_detected": len(
                key_info.get("legal_terms", [])
            ),
        },

        "disclaimer": (
            "This is an AI-generated summary for easier "
            "understanding. It should not be treated as "
            "legal advice."
        ),
    }


# ============================================================
# STRUCTURED SUMMARY (legacy flat text — kept for callers
# that still need a single printable string)
# ============================================================

def build_summary(
    bart_summary,
    key_info,
    doc_type,
    stats
):
    """
    Combine BART-generated natural summary with
    deterministic legal information.
    """

    parts = []

    parts.append(
        "LEGAL DOCUMENT SUMMARY"
    )

    parts.append(
        "======================"
    )

    parts.append("")

    # --------------------------------------------------------
    # DOCUMENT TYPE
    # --------------------------------------------------------

    parts.append(
        "DOCUMENT TYPE"
    )

    parts.append(
        "------------"
    )

    parts.append(
        doc_type
    )

    parts.append("")

    # --------------------------------------------------------
    # AI SUMMARY
    # --------------------------------------------------------

    parts.append(
        "SUMMARY"
    )

    parts.append(
        "-------"
    )

    if bart_summary:

        parts.append(
            bart_summary
        )

    else:

        parts.append(
            "A summary could not be generated."
        )

    parts.append("")

    # --------------------------------------------------------
    # PARTIES
    # --------------------------------------------------------

    if key_info["parties"]:

        parts.append(
            "PARTIES"
        )

        parts.append(
            "-------"
        )

        for party in key_info["parties"]:

            parts.append(
                f"• {party}"
            )

        parts.append("")

    # --------------------------------------------------------
    # DATES
    # --------------------------------------------------------

    if key_info["dates"]:

        parts.append(
            "IMPORTANT DATES"
        )

        parts.append(
            "---------------"
        )

        for date in key_info["dates"]:

            parts.append(
                f"• {date}"
            )

        parts.append("")

    # --------------------------------------------------------
    # FINANCIAL TERMS
    # --------------------------------------------------------

    if key_info["amounts"]:

        parts.append(
            "FINANCIAL TERMS"
        )

        parts.append(
            "---------------"
        )

        for amount in key_info["amounts"]:

            parts.append(
                f"• {amount}"
            )

        parts.append("")

    # --------------------------------------------------------
    # KEY TERMS / OBLIGATIONS
    # --------------------------------------------------------

    if key_info["key_points"]:

        parts.append(
            "KEY TERMS AND OBLIGATIONS"
        )

        parts.append(
            "-------------------------"
        )

        for point in key_info["key_points"]:

            point = point.strip()

            if point:

                parts.append(
                    f"• {point}"
                )

        parts.append("")

    # --------------------------------------------------------
    # IMPORTANT CLAUSES
    # --------------------------------------------------------

    if key_info["clauses"]:

        parts.append(
            "IMPORTANT CLAUSES"
        )

        parts.append(
            "-----------------"
        )

        for clause in key_info["clauses"]:

            parts.append(
                f"• {clause}"
            )

        parts.append("")

    # --------------------------------------------------------
    # LEGAL TERMS
    # --------------------------------------------------------

    if key_info.get(
        "legal_terms"
    ):

        parts.append(
            "KEY LEGAL TERMS"
        )

        parts.append(
            "----------------"
        )

        parts.append(
            "• "
            + ", ".join(
                key_info["legal_terms"]
            )
        )

        parts.append("")

    # --------------------------------------------------------
    # DOCUMENT STATISTICS
    # --------------------------------------------------------

    parts.append(
        "DOCUMENT STATISTICS"
    )

    parts.append(
        "-------------------"
    )

    parts.append(
        f"• Words: "
        f"{stats.get('total_words', 0)}"
    )

    parts.append(
        f"• Sentences: "
        f"{stats.get('total_sentences', 0)}"
    )

    parts.append(
        f"• Legal terms detected: "
        f"{len(key_info.get('legal_terms', []))}"
    )

    parts.append("")

    parts.append(
        "Note: This is an AI-generated summary "
        "for easier understanding. It should not "
        "be treated as legal advice."
    )

    return "\n".join(
        parts
    )


# ============================================================
# STRUCTURED (JSON) SUMMARY — PROFESSIONAL OUTPUT
# ============================================================

def build_structured_summary(
    bart_summary,
    key_info,
    doc_type,
    stats
):
    """
    Build a clean, structured summary object for the
    frontend/API instead of a flat text blob.

    Every section is its own field so the UI can render
    cards/lists directly without parsing text.
    """

    return {

        "document_type": doc_type,

        "ai_summary": (
            bart_summary
            if bart_summary
            else "A summary could not be generated."
        ),

        "parties": key_info.get("parties", []),

        "important_dates": key_info.get("dates", []),

        "financial_terms": key_info.get("amounts", []),

        "key_terms_and_obligations": key_info.get(
            "key_points", []
        ),

        "important_clauses": key_info.get("clauses", []),

        "key_legal_terms": key_info.get("legal_terms", []),

        "statistics": {
            "total_words": stats.get("total_words", 0),
            "total_sentences": stats.get(
                "total_sentences", 0
            ),
            "legal_terms_detected": len(
                key_info.get("legal_terms", [])
            ),
        },

        "formatted_text": build_summary(
            bart_summary,
            key_info,
            doc_type,
            stats
        ),

        "disclaimer": (
            "This is an AI-generated summary for easier "
            "understanding. It should not be treated as "
            "legal advice."
        ),
    }


def _empty_summary(message: str) -> dict:
    """
    Consistent structured shape even when summarization
    cannot run, so the frontend never has to branch on
    string-vs-object.
    """

    return {
        "document_type": "Unknown",
        "ai_summary": message,
        "parties": [],
        "important_dates": [],
        "financial_terms": [],
        "key_terms_and_obligations": [],
        "important_clauses": [],
        "key_legal_terms": [],
        "statistics": {
            "total_words": 0,
            "total_sentences": 0,
            "legal_terms_detected": 0,
        },
        "formatted_text": message,
        "disclaimer": (
            "This is an AI-generated summary for easier "
            "understanding. It should not be treated as "
            "legal advice."
        ),
    }


# ============================================================
# MAIN SUMMARIZATION FUNCTION
# ============================================================

def summarize_document(
    cleaned_text,
    stats,
    top_n=8
):
    """
    Main summarization function.

    BART is used for the natural-language summary.

    Deterministic extraction is retained for:
        - parties
        - dates
        - financial terms
        - important clauses
        - key legal terms
        - document statistics

    This gives both:
        1. Human-readable summary
        2. Structured factual information
    """

    print(
        "\n[Summarizer] "
        "Starting BART summarization..."
    )

    cleaned_text = clean_text(
        cleaned_text
    )

    if not cleaned_text:

        return _empty_summary(
            "Document is empty and cannot be summarized."
        )

    word_count = len(
        cleaned_text.split()
    )

    if word_count < 20:

        return _empty_summary(
            "Document is too short to summarize."
        )

    # --------------------------------------------------------
    # Extract deterministic information
    # --------------------------------------------------------

    doc_type = detect_document_type(
        cleaned_text
    )

    key_info = extract_key_info(
        cleaned_text
    )

    print(
        f"[Summarizer] "
        f"Detected type: {doc_type}"
    )

    # --------------------------------------------------------
    # BART summary
    # --------------------------------------------------------

    try:

        bart_summary = (
            generate_bart_summary(
                cleaned_text
            )
        )

    except Exception as e:

        print(
            "[Summarizer] BART failed:"
            f" {e}"
        )

        # Graceful fallback.
        # If BART fails, return a compact
        # deterministic summary instead of
        # crashing the complete legal pipeline.

        sentences = re.split(
            r"(?<=[.;])\s+",
            cleaned_text
        )

        sentences = [
            clean_text(s)
            for s in sentences
            if len(
                clean_text(s).split()
            ) >= 5
        ]

        bart_summary = " ".join(
            sentences[:top_n]
        )

    summary = build_structured_summary(
        bart_summary,
        key_info,
        doc_type,
        stats
    )

    print(
        "[Summarizer] Done ✓"
    )

    return summary