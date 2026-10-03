# 🏛️ AI Legal Document Simplifier

> An intelligent legal document risk analyzer that automatically detects risky clauses, generates safer alternatives, and provides negotiation advice — powered by hybrid AI.

---

## 📌 What It Does

- Upload any legal PDF (Employment Agreement, NDA, Power of Attorney, Sale Deed, Service Contract)
- AI extracts and analyzes every clause
- Each clause is classified as **HIGH**, **MEDIUM**, or **LOW** risk
- Generates **safer rewritten clauses** in formal legal language
- Detects **missing important clauses**
- Provides **negotiation advice**
- Returns a structured JSON report for frontend dashboard

---

## 🏗️ System Architecture

```
PDF Upload / Clause Input
        ↓
Document Loader     → Text extraction + Tesseract OCR (for scanned PDFs)
        ↓
Document Cleaner    → Remove noise, headers, footers
        ↓
Clause Extractor    → Split into individual legal clauses
        ↓
Risk Classifier     → Hybrid: Rule-based keywords + Semantic LLM classifier
        ↓
FAISS RAG           → Retrieve similar clauses from legal database
        ↓
LLM Reasoner        → Generate reason, safer clause, negotiation advice
        ↓
Missing Clause Detector → Flag absent critical clauses
        ↓
Report Generator    → JSON dashboard report + Full report saved
        ↓
FastAPI REST API    → Frontend-ready response
```

---

## 🤖 Two AI Modes

### ☁️ Mode 1 — Cloud LLM (Groq API) — Recommended
- Uses **Llama 3.1 8B Instant** via Groq
- Free API — no GPU needed
- Fast — ~2-3 seconds per clause
- Works on any laptop/PC

### 🖥️ Mode 2 — Local LLM (LM Studio)
- Runs **Mistral 7B** or **Qwen 2.5 7B** locally
- No internet needed after download
- Requires LM Studio running on port `1234`
- Good for offline/private use

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend API | FastAPI |
| Server | Uvicorn |
| PDF Text Extraction | PyMuPDF (fitz) |
| OCR (Scanned PDFs) | Tesseract OCR + pytesseract |
| Image Processing | Pillow |
| Cloud LLM | Groq API (Llama 3.1 8B) |
| Local LLM | LM Studio (Mistral 7B / Qwen 2.5) |
| Risk Classification | Hybrid Rule-based + Semantic LLM |
| Vector Database | FAISS |
| Embeddings | Sentence Transformers (BGE) |
| Configuration | python-dotenv |
| Language | Python 3.11+ |

---

## 📋 Prerequisites

### Required Software

| Software | Download Link | Required For |
|----------|--------------|-------------|
| Python 3.11+ | https://www.python.org/downloads/ | Backend |
| Tesseract OCR | https://github.com/UB-Mannheim/tesseract/wiki | Scanned PDFs |
| Git | https://git-scm.com/ | Clone repo |

### For Cloud Mode (Groq)
- Free account at https://console.groq.com
- Create API key → copy it

### For Local Mode (LM Studio)
- Download LM Studio from https://lmstudio.ai
- Download model: `mistral-7b-instruct-v0.2.Q4_K_M.gguf` (~4GB)
- Start Local Server on port `1234`

---

## 🚀 Setup & Installation

### Step 1 — Clone the repository

```bash
git clone https://github.com/chanyakyapadole08/AI-Legal-Document-Simplifier.git
cd AI-Legal-Document-Simplifier
```

### Step 2 — Create virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Install Tesseract OCR

**Windows:**
1. Download installer from: https://github.com/UB-Mannheim/tesseract/wiki
2. Install to default path: `C:\Program Files\Tesseract-OCR`

**Mac:**
```bash
brew install tesseract
```

**Linux:**
```bash
sudo apt install tesseract-ocr
```

### Step 5 — Configure environment

```bash
# Copy example env file
copy .env.example .env    # Windows
cp .env.example .env      # Mac/Linux
```

Edit `.env` file:

```env
# ☁️ FOR CLOUD MODE (Groq) — Recommended
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_API_KEY=your_groq_api_key_here
LLM_MODEL_NAME=llama-3.1-8b-instant

# 🖥️ FOR LOCAL MODE (LM Studio)
# LLM_BASE_URL=http://localhost:1234/v1
# LLM_API_KEY=lm-studio
# LLM_MODEL_NAME=mistral-7b-instruct-v0.2

# Settings
ENABLE_RAG=true
ENABLE_SEMANTIC_RISK=true
SEMANTIC_RISK_MODE=fast
MAX_ANALYSIS_CLAUSES=30
```

### Step 6 — Run the backend

```bash
python -m uvicorn services.api:app --reload
```

Server starts at: **http://127.0.0.1:8000**

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API info and all endpoints |
| GET | `/health` | Health check |
| POST | `/analyze/clause` | Analyze single clause |
| POST | `/analyze/pdf` | Upload and analyze PDF |
| POST | `/agent/clause` | Alias for clause analysis |
| POST | `/agent/pdf` | Alias for PDF analysis |
| GET | `/reports/{file_name}` | Download saved report |
| GET | `/docs` | Swagger UI (interactive testing) |

### Test via Swagger UI
Open: **http://127.0.0.1:8000/docs**

### Test via curl

**Single clause:**
```bash
curl -X POST "http://127.0.0.1:8000/analyze/clause" \
  -H "Content-Type: application/json" \
  -d '{"clause": "The Employer may terminate this Agreement at any time without notice."}'
```

**PDF upload:**
```bash
curl -X POST "http://127.0.0.1:8000/analyze/pdf" \
  -F "file=@your_contract.pdf"
```

---

## 📊 Sample Output

```json
{
  "risk_level": "HIGH",
  "risk_score": 90,
  "category": "Termination without notice",
  "detected_issue": "Termination without notice",
  "reason": "This clause allows the Employer to terminate employment at any time without providing any notice period, leaving the Employee with no time to prepare or seek alternative employment.",
  "action": "Negotiate a minimum 30-day written notice period before termination.",
  "safer_clause": "The Employer may terminate this Agreement only upon providing at least thirty (30) days prior written notice specifying the grounds for termination, except in cases of proven gross misconduct.",
  "negotiation_advice": "Request a mutual notice period clause and severance pay provision in cases of termination without cause."
}
```

---

## 📁 Project Structure

```
AI-Legal-Document-Simplifier/
├── agent/
│   └── legal_agent.py          # Agent orchestrator
├── services/
│   ├── api.py                  # FastAPI endpoints
│   ├── analyzer_v2.py          # Main analysis pipeline
│   ├── clause_extractor.py     # Clause splitting
│   ├── document_loader.py      # PDF + OCR extraction
│   ├── document_cleaner.py     # Text cleaning
│   ├── risk_classifier.py      # Hybrid risk classifier
│   ├── semantic_risk_classifier.py  # LLM semantic classifier
│   ├── llm_reasoner.py         # LLM reasoning + safer clause
│   ├── vector_store.py         # FAISS RAG
│   ├── embedder.py             # Sentence embeddings
│   ├── missing_clause_detector.py  # Missing clause detection
│   ├── negotiation_engine.py   # Negotiation advice
│   ├── risk_score.py           # Document risk scoring
│   ├── simple_report_generator.py  # Dashboard report
│   ├── json_report_writer.py   # Full JSON report
│   └── config.py               # Configuration
├── datasets/                   # Legal clause datasets
├── vector_store/               # FAISS index files
├── reports/                    # Generated reports
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🔧 Switch Between Cloud and Local Mode

### Cloud Mode (Groq) — `.env`
```env
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_API_KEY=your_groq_api_key_here
LLM_MODEL_NAME=llama-3.1-8b-instant
```

### Local Mode (LM Studio) — `.env`
```env
LLM_BASE_URL=http://localhost:1234/v1
LLM_API_KEY=lm-studio
LLM_MODEL_NAME=mistral-7b-instruct-v0.2
```

> **Note:** For Local Mode, open LM Studio → Load model → Start Local Server on port 1234

---

## 📦 Requirements

```
fastapi==0.115.6
uvicorn[standard]==0.34.0
python-multipart==0.0.20
python-dotenv==1.0.1
pydantic==2.10.4
openai==1.59.7
PyMuPDF==1.25.1
sentence-transformers==3.3.1
faiss-cpu==1.9.0.post1
numpy==1.26.4
scikit-learn==1.6.0
requests==2.32.3
```

---

## 👨‍💻 Developer

**Chanyakya Padole**
- GitHub: [@chanyakyapadole08](https://github.com/chanyakyapadole08)
- Project: AI Legal Document Simplifier — Final Year Project

---

## 📄 License

This project is for educational and research purposes.
