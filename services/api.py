from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from services.summarizer import summarize_document
from services.translator import run_translation_pipeline
from services.plain_language import build_plain_translations, translate_text
import shutil
import os
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
from services.summarizer import summarize_document
from services.analyzer_v2 import analyze_document_v2, analyze_clause_v2
from services.json_report_writer import save_json_report
from services.simple_report_generator import (
    build_simple_report,
    simplify_clause_result
)
from services.config import settings


app = FastAPI(
    title="AI Legal Risk Analyzer V2",
    description="Hybrid legal document risk analyzer using classifier, semantic risk, FAISS RAG and configurable LLM",
    version="2.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ClauseInput(BaseModel):
    clause: str


@app.get("/")
def home():
    return {
        "message": "AI Legal Risk Analyzer V2 Running",
        "architecture": {
            "risk_classification": "Hybrid rule-based + semantic classifier",
            "retrieval": "FAISS RAG",
            "llm_role": "Reasoning and safer clause generation only",
           "summary_module": "Professional structured summary",
            "translation_module": "English, Hindi and Marathi translation",
            "frontend_output": "Professional dashboard report",
            "full_report": "Saved in reports folder"
        },
        "endpoints": {
            "clause": "POST /analyze/clause",
            "clause_alias": "POST /agent/clause",
            "pdf": "POST /analyze/pdf",
            "pdf_alias": "POST /agent/pdf",
            "download_report": "GET /reports/{file_name}",
            "health": "GET /health",
            "docs": "GET /docs"
        }
    }


@app.post("/analyze/clause")
def analyze_clause_endpoint(data: ClauseInput):
    if not data.clause.strip():
        raise HTTPException(
            status_code=400,
            detail="Clause cannot be empty"
        )

    full_result = analyze_clause_v2(
        data.clause
    )

    simple_result = simplify_clause_result(
        full_result
    )

    if settings.RETURN_FULL_REPORT:
        return {
            "status": "success",
            "result": simple_result,
            "full_result": full_result
        }

    return {
        "status": "success",
        "result": simple_result
    }


@app.post("/agent/clause")
def analyze_clause_agent_alias(data: ClauseInput):
    return analyze_clause_endpoint(data)

@app.post("/analyze/pdf")
async def analyze_pdf_endpoint(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed"
        )

    os.makedirs("temp", exist_ok=True)
    os.makedirs("processed_documents", exist_ok=True)

    safe_filename = os.path.basename(file.filename)

    file_path = os.path.join(
        "temp",
        safe_filename
    )

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(
            file.file,
            buffer
        )

    try:
        # =====================================================
        # STEP 1 — RISK ANALYSIS
        # =====================================================
        full_report = analyze_document_v2(
            file_path
        )

        # Cleaned text produced by analyzer
        cleaned_text = full_report.get(
            "cleaned_text",
            ""
        )

        # =====================================================
        # STEP 2 — SUMMARY
        # =====================================================
        words = cleaned_text.split()

        sentences = [
            s.strip()
            for s in cleaned_text.replace(
                "!", "."
            ).replace(
                "?", "."
            ).split(".")
            if s.strip()
        ]

        stats = {
            "total_words": len(words),
            "total_sentences": len(sentences),
            "legal_terms_found": []
        }

        summary = summarize_document(
            cleaned_text,
            stats
        )

        # =====================================================
        # STEP 3 — TRANSLATION
        # =====================================================
        translation_results = build_plain_translations(
            summary,
            full_report,
            cleaned_text
        )
        translation_sources = translation_results.pop("_sources", {})

        translation = {
            "results": translation_results,
            "errors": []
        }
        # =====================================================
        # STEP 4 — ADD SUMMARY + TRANSLATION
        # =====================================================
        full_report["summary"] = summary

        full_report["translation"] = (
            translation["results"]
        )

        full_report["translation_errors"] = (
            translation["errors"]
        )

        # =====================================================
        # STEP 5 — SAVE JSON REPORT
        # =====================================================
        report_path = save_json_report(
            full_report
        )

        full_report["report_path"] = report_path

        # =====================================================
        # STEP 6 — BUILD FRONTEND REPORT
        # =====================================================
        simple_report = build_simple_report(
            full_report
        )

        # =====================================================
        # STEP 7 — SAVE PROCESSED DOCUMENT
        # =====================================================
        base_name = os.path.splitext(
            safe_filename
        )[0]

        processed_path = os.path.join(
            "processed_documents",
            base_name + "_processed.json"
        )

        import json

        with open(
            processed_path,
            "w",
            encoding="utf-8"
        ) as processed_file:

            json.dump(
                full_report,
                processed_file,
                ensure_ascii=False,
                indent=2
            )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Document processing failed: {str(e)}"
        )

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

    response = {
        "status": "success",
        "message": "Document analyzed, summarized and translated successfully",

        "document": {
            "original_filename": safe_filename,
            "processed_file": processed_path
        },

        "summary": summary,

        "translation": translation["results"],

        "translation_sources": translation_sources,

        "translation_errors": translation["errors"],

        "report": simple_report,

        "report_path": report_path
    }

    if settings.RETURN_FULL_REPORT:
        response["full_report"] = full_report

    return response

class PlainTranslateInput(BaseModel):
    text: str
    language: str


@app.post("/agent/translate")
def agent_translate(data: PlainTranslateInput):
    language = data.language.strip().lower()
    if language not in ("hindi", "marathi"):
        raise HTTPException(status_code=400, detail="Language must be hindi or marathi")
    if not data.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    result = translate_text(data.text, language)
    if not result:
        raise HTTPException(
            status_code=503,
            detail="The AI translation service is busy. Please try again in a minute."
        )
    return {"status": "success", "language": language, "text": result}


@app.post("/agent/pdf")
async def analyze_pdf_agent_alias(file: UploadFile = File(...)):
    return await analyze_pdf_endpoint(file)


@app.get("/reports/{file_name}")
def download_report(file_name: str):
    safe_name = os.path.basename(
        file_name
    )

    file_path = os.path.join(
        "reports",
        safe_name
    )

    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=404,
            detail="Report not found"
        )

    return FileResponse(
        file_path,
        media_type="application/json",
        filename=safe_name
    )


@app.post("/translate")
def translate_placeholder():
    return {
        "status": "not_available",
        "message": "Translation module is not integrated yet."
    }


@app.post("/summary")
def summary_placeholder():
    return {
        "status": "available",
        "message": "Summary is available through POST /analyze/pdf using the local BART model."
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": "2.0",

        "llm_primary": "Gemini",
        "llm_model": settings.LLM_MODEL_NAME,

        "llm_fallback": settings.LOCAL_LLM_MODEL,

        "risk_classifier": "LLM-first",

        "summary_model": settings.SUMMARY_MODEL_NAME,

        "semantic_risk": settings.ENABLE_SEMANTIC_RISK,

        "rag": settings.ENABLE_RAG,
        "rag_enabled": settings.ENABLE_RAG,

        "rag_backend": settings.RAG_BACKEND,

        "max_analysis_clauses": settings.MAX_ANALYSIS_CLAUSES
    }