import os
from pathlib import Path


try:
    from dotenv import load_dotenv

    BASE_DIR = Path(__file__).resolve().parent.parent
    ENV_PATH = BASE_DIR / ".env"

    load_dotenv(
        dotenv_path=ENV_PATH,
        override=True
    )

except Exception:
    pass


class Settings:
    """
    Central configuration for the AI Legal Document Analysis System.
    """

    LLM_BASE_URL = os.getenv(
        "LLM_BASE_URL",
        "https://generativelanguage.googleapis.com/v1beta/openai/"
    )

    LLM_API_KEY = os.getenv(
        "LLM_API_KEY",
        ""
    )

    LLM_MODEL_NAME = os.getenv(
        "LLM_MODEL_NAME",
        "gemini-3.8-flash"
    )

    LLM_TIMEOUT = float(
        os.getenv(
            "LLM_TIMEOUT",
            "30"
        )
    )

    # =====================================================
    # LOCAL QWEN FALLBACK
    # =====================================================

    LOCAL_LLM_MODEL = os.getenv(
        "LOCAL_LLM_MODEL",
        "Qwen/Qwen2.5-1.5B-Instruct"
    )

    ENABLE_SEMANTIC_RISK = os.getenv(
        "ENABLE_SEMANTIC_RISK",
        "false"
    ).lower() == "true"

    SEMANTIC_RISK_MODE = os.getenv(
        "SEMANTIC_RISK_MODE",
        "fast"
    )

    ENABLE_RAG = os.getenv(
        "ENABLE_RAG",
        "true"
    ).lower() == "true"

    RAG_BACKEND = os.getenv(
        "RAG_BACKEND",
        "faiss"
    )

    MAX_ANALYSIS_CLAUSES = int(
        os.getenv(
            "MAX_ANALYSIS_CLAUSES",
            "30"
        )
    )

    RETURN_FULL_REPORT = os.getenv(
        "RETURN_FULL_REPORT",
        "false"
    ).lower() == "true"
        # =====================================================
    # LOCAL BART SUMMARIZATION
    # =====================================================

    SUMMARY_MODEL_NAME = os.getenv(
        "SUMMARY_MODEL_NAME",
        "facebook/bart-large-cnn"
    )

    SUMMARY_MAX_INPUT_TOKENS = int(
        os.getenv(
            "SUMMARY_MAX_INPUT_TOKENS",
            "900"
        )
    )

    SUMMARY_MAX_LENGTH = int(
        os.getenv(
            "SUMMARY_MAX_LENGTH",
            "180"
        )
    )

    SUMMARY_MIN_LENGTH = int(
        os.getenv(
            "SUMMARY_MIN_LENGTH",
            "60"
        )
    )


settings = Settings()
