"""
Plain-language summary + translation (English / Hindi / Marathi).

One Gemini call per language turns the analysis into short, simply worded,
well formatted text (Markdown-lite: '## heading', '- bullet', '**bold**').
If Gemini fails for a language, the old machine translation is used instead.
"""
import os
import re
from concurrent.futures import ThreadPoolExecutor

from services import llm_cache
from services.llm_reasoner import MODEL_NAME, call_with_transient_retry, client
from services.translator import run_translation_pipeline

# Optional: in .env set PLAIN_LANG_MODELS=model-a,model-b to try other models
# when the main one is overloaded (503) or out of quota (429).
MODELS = [MODEL_NAME] + [m.strip() for m in os.getenv("PLAIN_LANG_MODELS", "").split(",") if m.strip()]

LANGS = {
    "english": "simple English",
    "hindi": "simple, everyday Hindi (सरल हिंदी) in Devanagari script",
    "marathi": "simple, everyday Marathi (सोपी मराठी) in Devanagari script",
}

GLOSSARY = {
    "hindi": (
        "agreement=अनुबंध; employer=नियोक्ता; employee=कर्मचारी; termination=समाप्ति; notice period=नोटिस अवधि; "
        "probation=परिवीक्षा अवधि; confidentiality=गोपनीयता; non-compete=प्रतिस्पर्धा न करने की शर्त; "
        "non-solicitation=ग्राहकों या सहकर्मियों को न बुलाने की शर्त; indemnity=क्षतिपूर्ति; liability=दायित्व; "
        "intellectual property=बौद्धिक संपदा; arbitration=मध्यस्थता; jurisdiction=क्षेत्राधिकार; penalty=जुर्माना; "
        "bond=बॉन्ड; salary=वेतन; clause=धारा; breach=उल्लंघन; governing law=लागू कानून; dispute=विवाद; resignation=इस्तीफ़ा"
    ),
    "marathi": (
        "agreement=करार; employer=नियोक्ता; employee=कर्मचारी; termination=सेवा समाप्ती; notice period=नोटीस कालावधी; "
        "probation=परिविक्षा कालावधी; confidentiality=गोपनीयता; non-compete=स्पर्धा न करण्याची अट; "
        "non-solicitation=ग्राहक किंवा सहकाऱ्यांना न बोलावण्याची अट; indemnity=नुकसानभरपाई; liability=जबाबदारी; "
        "intellectual property=बौद्धिक संपदा; arbitration=लवाद; jurisdiction=अधिकारक्षेत्र; penalty=दंड; "
        "bond=बॉण्ड; salary=पगार; clause=कलम; breach=उल्लंघन; governing law=लागू कायदा; dispute=वाद; resignation=राजीनामा"
    ),
}

PROMPT = """You explain legal documents to ordinary people in India who have no legal training.

Rewrite the analysis below in {language}.

Rules:
- Use very simple, short sentences and common spoken words. Avoid heavy legal or formal words; explain any unavoidable legal term in a few simple words.
- Keep names of people/companies, dates, amounts and numbers exactly as in the source. Write numbers with ordinary digits 0-9.
- Do not add, remove or change any fact. If unsure of a legal term, explain it simply instead of guessing.
- Where these terms apply, use exactly these standard words: {glossary}
- Translate the section headings too.
- Format EXACTLY like this and nothing else:
  ## Heading
  - short bullet
  - short bullet
  A plain sentence line is allowed under the first heading for the overview.
- Use **bold** only for the most important words (amounts, dates, deadlines).
- Sections, in this order (skip a section if the source has nothing for it):
  1. What this document is (type + 2-3 line overview)
  2. Who is involved
  3. Important dates
  4. Money and payments
  5. Main duties and rights
  6. Risks to watch (most serious first, say why in one line each)
  7. Missing protections
- No tables, no code blocks, no introduction or closing remarks.

SOURCE:
{source}
"""


def _flat(v) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, dict):
        return " - ".join(_flat(x) for x in v.values() if x)
    if isinstance(v, (list, tuple)):
        return "; ".join(_flat(x) for x in v if x)
    return str(v)


def _clip(text, n=220) -> str:
    text = re.sub(r"\s+", " ", _flat(text)).strip(" -•*\t")
    if len(text) <= n:
        return text
    return text[:n].rsplit(" ", 1)[0] + "…"


def _items(v, limit=6, n=220):
    seq = v if isinstance(v, (list, tuple)) else ([v] if v else [])
    out, seen = [], set()
    for x in seq:
        s = _clip(x, n)
        k = s.lower()
        if s and k not in seen:
            seen.add(k)
            out.append(s)
        if len(out) >= limit:
            break
    return out


def build_source(summary: dict, full_report: dict) -> str:
    s = summary or {}
    lines = [f"Document type: {_clip(s.get('document_type'), 80)}",
             f"Overview: {_clip(s.get('ai_summary'), 900)}"]
    for title, key, limit in (
        ("Parties", "parties", 6),
        ("Important dates", "important_dates", 6),
        ("Money / payments", "financial_terms", 6),
        ("Key duties and terms", "key_terms_and_obligations", 6),
    ):
        items = _items(s.get(key), limit)
        if items:
            lines.append(f"{title}:")
            lines += [f"- {i}" for i in items]

    order = {"HIGH": 0, "MEDIUM": 1}
    risky = [c for c in (full_report or {}).get("clauses", [])
             if str(c.get("risk_level", "")).upper() in order]
    risky.sort(key=lambda c: order[str(c.get("risk_level")).upper()])
    if risky:
        lines.append("Risky clauses:")
        for c in risky[:6]:
            head = _clip(c.get("heading") or c.get("category"), 70)
            why = _clip(c.get("reason") or c.get("detected_issue"), 200)
            lines.append(f"- [{str(c.get('risk_level')).upper()}] {head}: {why}")

    missing = (full_report or {}).get("missing_clauses") or []
    mlines = []
    for m in missing[:5]:
        if isinstance(m, dict):
            name = _clip(m.get("missing_clause") or m.get("clause") or m.get("title") or m.get("name"), 80)
            why = _clip(m.get("why_important") or m.get("reason") or m.get("risk_if_missing"), 160)
            mlines.append(f"- {name}: {why}" if why else f"- {name}")
        else:
            mlines.append(f"- {_clip(m, 200)}")
    if mlines:
        lines.append("Missing or weak clauses:")
        lines += mlines
    return "\n".join(lines)


def _valid(text: str, lang: str) -> bool:
    if not text or len(text) < 80:
        return False
    if lang in ("hindi", "marathi"):
        dev = len(re.findall(r"[\u0900-\u097F]", text))
        return dev / max(len(text), 1) > 0.3
    return True


def _clean_output(text: str) -> str:
    text = re.sub(r"^```[a-z]*\s*|\s*```$", "", (text or "").strip())
    return text.strip()


_DEV = str.maketrans("०१२३४५६७८९", "0123456789")


def _numbers(text: str) -> set:
    found = re.findall(r"\d[\d,]*\.?\d*", str(text).translate(_DEV))
    return {n.strip(",.") for n in found if n.strip(",.")}


def _numbers_ok(source: str, out: str) -> bool:
    """Every amount/date/number in the source must still be in the translation."""
    src, got = _numbers(source), _numbers(out)
    missing = src - got
    return len(missing) <= (1 if len(src) > 8 else 0)


def _call_model(model: str, prompt: str, attempts: int, delay: int) -> str:
    resp = call_with_transient_retry(lambda: client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a careful legal-document translator and explainer. Output only the requested formatted text."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=4000,
        temperature=0.1,
    ), attempts=attempts, delay=delay)
    return _clean_output(resp.choices[0].message.content)


def _generate(lang: str, prompt: str, check_source: str):
    """Try each model; retry once when numbers went missing. Returns (text, verified)."""
    best = None
    for model in MODELS:
        for attempt in (1, 2):
            try:
                print(f"[PlainLang] Generating {lang} with {model} (try {attempt})...")
                text = _call_model(model, prompt, 4 if len(MODELS) == 1 else 3, 8 if len(MODELS) == 1 else 5)
            except Exception as e:
                print(f"[PlainLang] {lang} failed on {model}: {str(e)[:160]}")
                break  # next model
            if not _valid(text, lang):
                print(f"[PlainLang] {lang} output rejected (too short / wrong script)")
                continue
            if _numbers_ok(check_source, text):
                return text, True
            print(f"[PlainLang] {lang}: some numbers/dates missing, retrying...")
            best = best or text
    return best, False


def _gemini_one(lang: str, source: str):
    key = llm_cache.make_key("plain-v2", lang, source)
    cached = llm_cache.get(key)
    if cached and _valid(cached, lang):
        print(f"[PlainLang] {lang} served from cache")
        return cached
    prompt = PROMPT.format(language=LANGS[lang], source=source, glossary=GLOSSARY.get(lang, "(none)"))
    text, verified = _generate(lang, prompt, source)
    if text and verified:
        llm_cache.put_many([(key, text)])
    return text


TRANSLATE_PROMPT = """Translate this plain-language summary of a legal document into {language}.

Rules:
- Keep the structure exactly: the same '## ' headings, '- ' bullets and **bold** words, in the same order.
- Use very simple, short, everyday words that a person with no legal training understands.
- Do not add, remove or change any fact. Keep names, amounts, dates and numbers exactly (write numbers with ordinary digits 0-9).
- Where these terms apply, use exactly these standard words: {glossary}
- Output only the translation, no notes.

TEXT:
{text}
"""


def translate_text(text_en: str, lang: str):
    """Regenerate one language from the English plain-language text. Returns text or None."""
    if lang not in ("hindi", "marathi") or not str(text_en or "").strip():
        return None
    prompt = TRANSLATE_PROMPT.format(language=LANGS[lang], glossary=GLOSSARY[lang], text=text_en)
    text, verified = _generate(lang, prompt, text_en)
    if text and verified:
        llm_cache.put_many([(llm_cache.make_key("plain-v2", lang, text_en), text)])
    return text


def _formatted_fallback(lang: str, english_text: str):
    """Offline fallback: translate the formatted English line by line so
    headings, bullets and paragraphs are kept (no Gemini needed)."""
    from services import translator

    fn = getattr(translator, "_nllb_marathi" if lang == "marathi" else "_nllb_hindi", None)
    if fn is None:
        fn = translator.translate_to_marathi if lang == "marathi" else translator.translate_to_hindi

    out = []
    for raw in english_text.splitlines():
        line = raw.strip()
        if not line:
            out.append("")
            continue
        m = re.match(r"^(#{1,4}\s+|[-*\u2022]\s+)?(.*)$", line)
        prefix, body = m.group(1) or "", m.group(2).replace("**", "")
        try:
            tr = fn(body).strip() if body else ""
        except Exception as e:
            print(f"[PlainLang] line translation failed: {e}")
            tr = body
        out.append(f"{prefix.strip()} {tr}".strip() if prefix else tr)
    return "\n".join(out)


def build_plain_translations(summary: dict, full_report: dict, cleaned_text: str) -> dict:
    source = build_source(summary, full_report)
    langs = list(LANGS)
    with ThreadPoolExecutor(max_workers=3) as pool:
        got = dict(zip(langs, pool.map(lambda l: _gemini_one(l, source), langs)))

    result = {l: t for l, t in got.items() if t}
    sources = {l: "ai" for l in result}

    for lang in ("hindi", "marathi"):
        if lang not in result and result.get("english"):
            try:
                print(f"[PlainLang] Formatted offline translation for {lang}...")
                text = _formatted_fallback(lang, result["english"])
                if _valid(text, lang):
                    result[lang] = text
                    sources[lang] = "offline"
            except Exception as e:
                print(f"[PlainLang] Formatted fallback failed for {lang}: {e}")

    missing = [l for l in langs if l not in result]
    if missing:
        print(f"[PlainLang] Falling back to machine translation for: {missing}")
        try:
            for l, t in run_translation_pipeline(cleaned_text, missing).items():
                result[l] = t
                sources[l] = "offline"
        except Exception as e:
            print(f"[PlainLang] Fallback translation failed: {e}")
    result["_sources"] = sources
    return result
