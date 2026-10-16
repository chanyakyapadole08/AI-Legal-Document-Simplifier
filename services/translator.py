"""
LexiShield Translation Service

English -> Hindi:
LibreTranslate running locally through Docker.

English -> Marathi:
NLLB-200 distilled 600M model running locally.

English:
Returns cleaned English text.
"""

import os
import re
import requests
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)


# ============================================================
# CONFIGURATION
# ============================================================

LIBRETRANSLATE_URL = (
    "http://127.0.0.1:5100/translate"
)

MARATHI_MODEL_NAME = (
    "facebook/nllb-200-distilled-600M"
)

# Larger chunks = fewer model calls.
# 1200 is still kept reasonably safe for NLLB input limits.
MAX_CHARS_PER_CHUNK = 1200

# Process two Marathi chunks together.
# This can improve CPU throughput compared with
# translating every chunk separately.
MARATHI_BATCH_SIZE = 2

# NLLB tokenizer input limit.
MARATHI_MAX_INPUT_TOKENS = 384

# Maximum generated Marathi tokens per chunk.
MARATHI_MAX_NEW_TOKENS = 220


# ============================================================
# CPU OPTIMIZATION
# ============================================================

CPU_THREADS = max(
    1,
    min(
        8,
        os.cpu_count() or 4
    )
)

if not torch.cuda.is_available():

    try:
        torch.set_num_threads(
            CPU_THREADS
        )
    except Exception:
        pass


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    f"[Translator] Device: {DEVICE}"
)

if DEVICE.type == "cpu":

    print(
        f"[Translator] CPU threads: "
        f"{CPU_THREADS}"
    )

else:

    print(
        "[Translator] GPU acceleration enabled."
    )


# ============================================================
# HTTP SESSION
# ============================================================

# Reuse one HTTP connection for Hindi translation.
# This avoids creating a new connection for every chunk.
_http_session = requests.Session()


# ============================================================
# MARATHI MODEL
# ============================================================

_marathi_tokenizer = None
_marathi_model = None
_marathi_bos_token_id = None


def load_marathi_model():

    global _marathi_tokenizer
    global _marathi_model
    global _marathi_bos_token_id

    if (
        _marathi_tokenizer is not None
        and _marathi_model is not None
    ):
        return

    print(
        "Loading Marathi translation model..."
    )

    _marathi_tokenizer = (
        AutoTokenizer.from_pretrained(
            MARATHI_MODEL_NAME
        )
    )

    _marathi_model = (
        AutoModelForSeq2SeqLM.from_pretrained(
            MARATHI_MODEL_NAME
        )
    )

    # Move model only once.
    _marathi_model.to(
        DEVICE
    )

    # Evaluation mode.
    _marathi_model.eval()

    # English source language.
    _marathi_tokenizer.src_lang = (
        "eng_Latn"
    )

    # Marathi target language token.
    _marathi_bos_token_id = (
        _marathi_tokenizer
        .convert_tokens_to_ids(
            "mar_Deva"
        )
    )

    print(
        "Marathi translation model loaded."
    )


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(
    text: str
) -> str:

    if not text:
        return ""

    text = text.replace(
        "\r",
        "\n"
    )

    # Fix common PDF line-break problems.
    text = re.sub(
        r"(?<=[a-zA-Z,;:])\n(?=[a-zA-Z])",
        " ",
        text
    )

    # Remove excessive spaces.
    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    # Remove excessive blank lines.
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    return text.strip()


# ============================================================
# CHUNKING
# ============================================================

def split_text_into_chunks(
    text: str,
    max_chars: int = MAX_CHARS_PER_CHUNK
):

    text = clean_text(
        text
    )

    if not text:
        return []

    paragraphs = re.split(
        r"\n\s*\n",
        text
    )

    chunks = []

    current = ""

    for paragraph in paragraphs:

        paragraph = paragraph.strip()

        if not paragraph:
            continue

        candidate = (
            paragraph
            if not current
            else current
            + "\n\n"
            + paragraph
        )

        if len(candidate) <= max_chars:

            current = candidate

        else:

            if current:

                chunks.append(
                    current
                )

            # Very long paragraph.
            if len(paragraph) > max_chars:

                sentences = re.split(
                    r"(?<=[.!?;])\s+",
                    paragraph
                )

                current = ""

                for sentence in sentences:

                    sentence = (
                        sentence.strip()
                    )

                    if not sentence:
                        continue

                    candidate = (
                        sentence
                        if not current
                        else current
                        + " "
                        + sentence
                    )

                    if (
                        len(candidate)
                        <= max_chars
                    ):

                        current = candidate

                    else:

                        if current:

                            chunks.append(
                                current
                            )

                        current = sentence

            else:

                current = paragraph

    if current:

        chunks.append(
            current
        )

    return chunks


# ============================================================
# HINDI — LIBRETRANSLATE
# ============================================================

def translate_to_hindi(
    text: str
) -> str:

    chunks = split_text_into_chunks(
        text
    )

    if not chunks:
        return ""

    translated_chunks = []

    for index, chunk in enumerate(
        chunks
    ):

        try:

            response = (
                _http_session.post(
                    LIBRETRANSLATE_URL,
                    json={
                        "q": chunk,
                        "source": "en",
                        "target": "hi",
                        "format": "text"
                    },
                    timeout=120
                )
            )

            response.raise_for_status()

            data = response.json()

            translated = data.get(
                "translatedText",
                ""
            )

            if translated:

                translated_chunks.append(
                    translated.strip()
                )

            else:

                translated_chunks.append(
                    chunk
                )

            print(
                f"Hindi chunk "
                f"{index + 1}/"
                f"{len(chunks)} translated."
            )

        except Exception as exc:

            print(
                f"Hindi translation failed "
                f"for chunk "
                f"{index + 1}: {exc}"
            )

            # Safe fallback.
            translated_chunks.append(
                chunk
            )

    return "\n\n".join(
        translated_chunks
    )


# ============================================================
# MARATHI — NLLB
# ============================================================

def translate_to_marathi(
    text: str
) -> str:

    load_marathi_model()

    chunks = split_text_into_chunks(
        text
    )

    if not chunks:
        return ""

    translated_chunks = []

    total_chunks = len(
        chunks
    )

    # --------------------------------------------------------
    # Process chunks in small batches.
    # --------------------------------------------------------

    for batch_start in range(
        0,
        total_chunks,
        MARATHI_BATCH_SIZE
    ):

        batch = chunks[
            batch_start:
            batch_start
            + MARATHI_BATCH_SIZE
        ]

        try:

            # Make sure source language is English.
            _marathi_tokenizer.src_lang = (
                "eng_Latn"
            )

            # Tokenize multiple chunks together.
            inputs = _marathi_tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=(
                    MARATHI_MAX_INPUT_TOKENS
                )
            )

            # Move tensors to CPU/GPU.
            inputs = {
                key: value.to(DEVICE)
                for key, value in inputs.items()
            }

            # ------------------------------------------------
            # Fast inference.
            #
            # num_beams=1 = greedy decoding.
            # Much faster than beam search on CPU.
            # ------------------------------------------------

            with torch.inference_mode():

                generated = (
                    _marathi_model.generate(
                        **inputs,

                        forced_bos_token_id=(
                            _marathi_bos_token_id
                        ),

                        max_new_tokens=(
                            MARATHI_MAX_NEW_TOKENS
                        ),

                        # Major CPU speed improvement.
                        num_beams=1,

                        # Deterministic translation.
                        do_sample=False
                    )
                )

            # Decode every generated translation.
            decoded = (
                _marathi_tokenizer.batch_decode(
                    generated,
                    skip_special_tokens=True
                )
            )

            # ------------------------------------------------
            # Preserve exact chunk order.
            # ------------------------------------------------

            for local_index, translated in enumerate(
                decoded
            ):

                translated = (
                    translated.strip()
                )

                if not translated:

                    # Safe fallback.
                    translated = batch[
                        local_index
                    ]

                translated_chunks.append(
                    translated
                )

                actual_index = (
                    batch_start
                    + local_index
                    + 1
                )

                print(
                    f"Marathi chunk "
                    f"{actual_index}/"
                    f"{total_chunks} translated."
                )

        except Exception as exc:

            print(
                f"Marathi batch "
                f"{batch_start + 1}-"
                f"{min(batch_start + MARATHI_BATCH_SIZE, total_chunks)}"
                f" failed: {exc}"
            )

            # ------------------------------------------------
            # If batch translation fails,
            # preserve original text for each chunk.
            # ------------------------------------------------

            for chunk in batch:

                translated_chunks.append(
                    chunk
                )

    return "\n\n".join(
        translated_chunks
    )


# ============================================================
# ENGLISH
# ============================================================

def translate_to_english(
    text: str
) -> str:

    return clean_text(
        text
    )


# ============================================================
# MAIN TRANSLATION PIPELINE
# ============================================================

def run_translation_pipeline(
    text: str,
    target_languages: list
) -> dict:

    text = clean_text(
        text
    )

    result = {}

    languages = [
        str(language).lower()
        for language in target_languages
    ]

    # --------------------------------------------------------
    # English
    # --------------------------------------------------------

    if (
        "english" in languages
        or "en" in languages
    ):

        result["english"] = (
            translate_to_english(
                text
            )
        )

    # --------------------------------------------------------
    # Hindi
    # --------------------------------------------------------

    if (
        "hindi" in languages
        or "hi" in languages
    ):

        result["hindi"] = (
            translate_to_hindi(
                text
            )
        )

    # --------------------------------------------------------
    # Marathi
    # --------------------------------------------------------

    if (
        "marathi" in languages
        or "mr" in languages
    ):

        result["marathi"] = (
            translate_to_marathi(
                text
            )
        )

    return result

# ============================================================
# GOOGLE TRANSLATE (primary) - paste at the VERY END of translator.py
# Old LibreTranslate / NLLB functions stay as automatic fallback.
# pip install deep-translator
# ============================================================
from concurrent.futures import ThreadPoolExecutor

_libre_hindi = translate_to_hindi
_nllb_marathi = translate_to_marathi


def _google_translate(text: str, target: str) -> str:
    from deep_translator import GoogleTranslator

    chunks = split_text_into_chunks(text)
    if not chunks:
        return ""

    def one(chunk):
        for _ in range(3):
            try:
                out = GoogleTranslator(source="en", target=target).translate(chunk)
                if out and out.strip():
                    return out
            except Exception:
                pass
        raise RuntimeError("Google translate chunk failed")

    with ThreadPoolExecutor(max_workers=4) as pool:      # keeps order
        parts = list(pool.map(one, chunks))
    print(f"[Translator] Google Translate ({target}) done: {len(parts)} chunks")
    return "\n\n".join(parts)


def _nllb_hindi(text: str) -> str:
    global _marathi_bos_token_id
    load_marathi_model()
    original = _marathi_bos_token_id
    _marathi_bos_token_id = _marathi_tokenizer.convert_tokens_to_ids("hin_Deva")
    try:
        return _nllb_marathi(text)
    finally:
        _marathi_bos_token_id = original


def translate_to_hindi(text: str) -> str:
    try:
        return _google_translate(text, "hi")
    except Exception as e:
        print(f"[Translator] Google failed ({e}); using NLLB")
        return _nllb_hindi(text)

def translate_to_marathi(text: str) -> str:
    try:
        return _google_translate(text, "mr")
    except Exception as e:
        print(f"[Translator] Google failed ({e}); using NLLB")
        return _nllb_marathi(text)