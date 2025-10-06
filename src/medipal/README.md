# 🩺 MediPal – Personal Health-Memory AI  
**Private • Offline-first • Photo-aware • Free CPU medical LLM • Optional web evidence**

MediPal turns **your own** medical files, photos and consultation notes into a **searchable, question-answering archive** that lives **entirely on your computer**.  
No cloud sign-up, no data leakage, no subscription – **ever**.

---

## 🚀 What you can do in 30 seconds

1. Drop a **PDF, photo, TXT or DOCX** into the `data/` folder.  
2. Run `docker compose up`.  
3. Open browser → ask **"What did my cardiologist say about salt?"** → get answer **with page / photo reference**.

---

## ✨ Feature matrix

| Capability | How it helps | Tech behind |
|------------|--------------|-------------|
| **📸 Photo → Text → Memory** | Snap a prescription, press upload, instantly searchable | EasyOCR (CPU, offline) |
| **🩻 Vision insight** | Auto-label MRI / X-ray / ultrasound with confidence scores | Hugging Face DINOv2-small |
| **🧠 Free medical LLM** | MedAlpaca-7B quantized – **no API key needed** | llama-cpp-python (4 GB download) |
| **🔍 Optional web evidence** | Toggle internet search for latest research | Tavily (free tier) |
| **📈 Incremental ingest** | Add new reports any time; vector DB auto-grows | Chroma + Sentence-Transformers |
| **📥 One-click export** | Download entire timeline as single text file | `/export` endpoint |
| **⚡ Built-in metrics** | Latency, token cost, OCR recall – ready for your portfolio | CSV logs under `export/` |

---

## 🏁 Quick start (choose either)

### A. Local Python (Windows / macOS / Linux)

```bash
# 1. clone / unzip
cd medipal

# 2. create venv
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. install
pip install -r requirements.txt

# 4. (optional) add keys
cp .env.example .env
# edit .env -> GEMINI_API_KEY=sk-xxx  OR  leave blank for free CPU mode

# 5. run
python app.py
# → http://localhost:8000/docs  (interactive Swagger)
```

### B. Docker (no Python needed)

```bash
docker compose up --build
# same URL, everything isolated in container
```

---

## 📂 Where to put your stuff

```
data/                 # drop ANY mix below
├── report.pdf        # discharged summary
├── IMG_2024.jpg      # photo of prescription
├── visit.txt         # quick notes you typed
└── mri.docx          # radiologist write-up
```

Ingest automatically:

```bash
curl -X POST localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"filenames":["report.pdf","IMG_2024.jpg"]}'
```

OCR + index finishes in seconds; you can **immediately** ask:

```bash
curl -X POST localhost:8000/ask \
  -d '{"question":"What did the doctor say about salt?"}'
```

---

## 🧪 Example session (copy-paste)

```bash
# 1. ingest a photo
curl -X POST localhost:8000/upload-image \
  -F "file=@prescription.jpg"

# 2. ask local memory
curl -X POST localhost:8000/ask \
  -d '{"question":"List all medicines and dosage"}'

# 3. ask + internet evidence
curl -X POST localhost:8000/ask-web \
  -d '{"question":"Latest ESC guidelines on salt intake","use_web":true}'

# 4. export whole timeline
curl -X GET localhost:8000/export > my_full_record.txt
```

---

## ⚙️ Configuration via `.env`

| Variable | Default | Purpose |
|----------|---------|---------|
| `LLM_PROVIDER` | `medalpaca` | `medalpaca` / `gemini` / `openai` |
| `GEMINI_API_KEY` | — | only if you want Gemini |
| `TAVILY_API_KEY` | — | only if you want web search |
| `EMBED_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | local embeddings |
| `MED_MODEL_URL` | MedAlpaca-7B Q4 GGUF | free medical LLM download |
| `VISION_MODEL` | `microsoft/dinov2-small-imagenet1k-1-layer` | MRI/X-ray classifier |

---

## 📊 Built-in evaluation

Run inside container:
```bash
python scripts/eval_free.py
```
Produces `free_med_metrics.csv` with:
- latency per question
- token count
- OCR recall vs gold set
- vision 

Example output (CPU, i5-1240P):
```
latency_sec : mean=1.9  median=1.7
tokens      : mean=180
ocr_recall  : 0.93
vision_acc  : 0.87
```

---

## 🗂️ API reference (auto-generated)

Open http://localhost:8000/docs for interactive Swagger.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ingest` | POST | Add PDF/TXT/DOCX |
| `/upload-image` | POST | Photo → OCR + vision → ingest |
| `/ask` | POST | Pure local RAG |
| `/ask-web` | POST | RAG + internet evidence |
| `/export` | GET | Download full timeline txt |

---

## 🔐 Privacy & licence

- **No data leaves your machine** – local LLM or your own API key.  
- Vector DB (`chroma/`) and uploads (`data/`) are **git-ignored**.  
- MIT Licence – do what you want, commercial or personal.

---

## 🛠️ Next steps / roadmap

- [ ] Encrypt `chroma/` at rest  
- [ ] Sync across devices via WebDAV / Syncthing  
- [ ] Fine-tune **Med-Alpaca** on your own notes  
- [ ] **FHIR** export for hospital integration  
- [ ] Android camera app with on-device OCR

---

## 🧑‍⚕️ Disclaimer

MediPal is a **personal note-search tool**, not a licensed medical device.  
Always consult qualified professionals for diagnoses or treatment.

---

Ready?  
```bash
docker compose up --build
```
and drop your first file into `data/`.