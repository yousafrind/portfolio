
# 📊 PandasAI Streamlit Agent

> **Chat with your CSVs using a local LLM.**  
> Upload any dataset, type a natural-language question, and get smart analysis, charts, and summaries — powered by [PandasAI](https://github.com/gventuri/pandas-ai) and your **local Llama3 model** (via Ollama).

---

![Streamlit](https://img.shields.io/badge/Streamlit-Data_App-red?style=flat-square&logo=streamlit)
![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![Docker](https://img.shields.io/badge/Docker-Ready-informational?style=flat-square&logo=docker)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Local LLM](https://img.shields.io/badge/Local_LLM-Llama3_@_Ollama-lightgrey?style=flat-square&logo=ollama)

---

## 🧠 Overview

**PandasAI Streamlit Agent** lets you:
- Upload a `.csv` file
- Interact with it using **natural language**
- Get intelligent **summaries, calculations, and visualizations**
- Use **a local model (like Llama3)** instead of cloud APIs — keeping your data **secure and private**

This project demonstrates your ability to:
- Integrate LLMs into data apps
- Use modular software architecture
- Build and deploy modern Python dashboards

---

## ⚙️ Tech Stack

| Layer | Component | Purpose |
|:------|:-----------|:---------|
| 💬 LLM | [Ollama](https://ollama.ai) + Llama3 | Local model for reasoning |
| 🧮 Agent Framework | [PandasAI](https://github.com/gventuri/pandas-ai) | Natural language → Pandas code |
| 🧰 Backend | Python 3.11 | Core logic & modular architecture |
| 🖥️ Frontend | Streamlit | Web-based interactive UI |
| 🐳 DevOps | Docker | Portable containerized environment |
| ⚙️ Config | YAML | Easily switch model and settings |

---

## 🧩 Architecture

```text
                    ┌────────────────────────────┐
                    │        User Interface      │
                    │      (Streamlit Frontend)  │
                    └─────────────┬──────────────┘
                                  │
                                  ▼
                    ┌────────────────────────────┐
                    │       PandasAI Agent       │
                    │ (Converts NL to Pandas ops)│
                    └─────────────┬──────────────┘
                                  │
                                  ▼
                    ┌────────────────────────────┐
                    │        Local LLM API       │
                    │   (Ollama + Llama3 model)  │
                    └─────────────┬──────────────┘
                                  │
                                  ▼
                    ┌────────────────────────────┐
                    │         Data Layer         │
                    │     CSV / Pandas DataFrame │
                    └────────────────────────────┘
````

---

## 🚀 Features

| Feature                     | Description                                        | Status |
| --------------------------- | -------------------------------------------------- | ------ |
| 📂 File Upload              | Upload `.csv` files via Streamlit sidebar          | ✅      |
| 🧠 Natural Language Queries | Ask “What’s the average sales by region?”          | ✅      |
| 📊 Smart Visualizations     | Auto-plots data correlations and bar charts        | ✅      |
| 🧰 Modular Codebase         | Organized into `/modules` and `/config`            | ✅      |
| 🧾 YAML Config              | Easy model switching (Llama3, Phi3, Mistral, etc.) | ✅      |
| 🐳 Docker Support           | Fully containerized app for deployment             | ✅      |
| ⚙️ Logging                  | Structured logging and error handling              | ✅      |
| 💾 Caching                  | File caching via Streamlit for fast reloads        | ✅      |

---

## 🧮 Dummy Visualization Scenario

**Example Dataset:**
`retail_sales.csv`

| Date       | Region | Sales  | Units | Profit |
| ---------- | ------ | ------ | ----- | ------ |
| 2025-01-01 | North  | 12,500 | 430   | 2,350  |
| 2025-01-02 | South  | 9,800  | 290   | 1,800  |
| 2025-01-03 | East   | 14,200 | 510   | 2,750  |
| ...        | ...    | ...    | ...   | ...    |

---

### 🧠 Example Query

> **Prompt:** “Show me the total sales by region and visualize it.”

### 💡 Expected LLM Reasoning (simplified)

```
1. Group data by 'Region'
2. Sum the 'Sales' column
3. Plot a bar chart
```

### 📈 Output Visualization (Example)

```
North  | █████████████████████  12.5k  
South  | ████████████           9.8k  
East   | ████████████████████   14.2k  
```

*(In the actual Streamlit app, this would appear as a bar chart.)*

---

## 🖥️ Screenshots (Placeholder)

| Upload Data                         | Chat Interface                  | Visualization                     |
| ----------------------------------- | ------------------------------- | --------------------------------- |
| ![upload](assets/sample_upload.png) | ![chat](assets/sample_chat.png) | ![chart](assets/sample_chart.png) |

---

## ⚙️ Installation

### 🧾 Prerequisites

* Python ≥ 3.11
* [Ollama](https://ollama.ai) installed locally
* Model pulled:

  ```bash
  ollama pull llama3
  ```

### 🧰 Setup

```bash
git clone https://github.com/<your-username>/pandasai-streamlit-agent.git
cd pandasai-streamlit-agent
pip install -r requirements.txt
```

Then run:

```bash
streamlit run app.py
```

Visit [http://localhost:8501](http://localhost:8501).

---

## 🐳 Run with Docker

```bash
docker build -t pandasai-agent .
docker run -p 8501:8501 pandasai-agent
```

---

## ⚙️ Configuration

Edit `config/settings.yaml`:

```yaml
llm:
  api_base: "http://localhost:11434/v1"
  model: "llama3"
```

You can swap models easily:

```yaml
model: "phi3"
```

---

## 🧠 Example Prompt Ideas

| Prompt                                                 | Response Example                              |
| ------------------------------------------------------ | --------------------------------------------- |
| “What’s the trend of profit over time?”                | Generates a line chart of `Profit` vs. `Date` |
| “Compare average sales between North and East.”        | Returns numeric difference + bar chart        |
| “Show correlation matrix between all numeric columns.” | Displays a heatmap of correlations            |
| “Summarize this dataset in two sentences.”             | Natural-language description of key stats     |

---

## 🧩 Folder Structure

```
pandasai-streamlit-agent/
│
├── app.py                        # Main Streamlit app
├── config/settings.yaml           # Model & app settings
├── modules/                       # Modular Python logic
│   ├── llm_connector.py
│   ├── data_agent.py
│   ├── visualizer.py
│   ├── utils.py
│   └── __init__.py
├── assets/                        # Images & logos
├── requirements.txt
├── Dockerfile
├── README.md
└── .gitignore
```

---

## 📦 Example Requirements

```
streamlit>=1.37.0
pandas
pandasai
pyyaml
```

---

## 🤝 Contributing

Contributions are welcome!
Fork the repo, make your feature branch, and submit a pull request:

```bash
git checkout -b feature/my-awesome-feature
```

---

## 📄 License

MIT License © 2025 Yousaf Murtaza Rind

---

## 🌐 Future Roadmap

| Milestone                | Description                   | ETA      |
| ------------------------ | ----------------------------- | -------- |
| 🧩 Memory Integration    | Store conversation history    | Nov 2025 |
| 🔐 Authentication        | Add basic auth for multi-user | Dec 2025 |
| 🧮 Plotly Charts         | Add interactive charting      | Jan 2026 |
| 🧱 LangChain Integration | Multi-tool reasoning          | Feb 2026 |

---

## 🧭 Demo Flow Summary

1. **Upload CSV** → `retail_sales.csv`
2. **App detects columns** → auto summary
3. **User types prompt:** “Plot total sales by region”
4. **LLM generates Python code** via PandasAI
5. **Agent executes it** on the dataframe
6. **Streamlit displays** chart + textual summary

---

### ⭐ Show your Support

If this repo helped you or you want to showcase your LLM engineering skills:

* Give it a ⭐ on GitHub
* Add it to your portfolio
* Share it on LinkedIn with the tag **#AIEngineering**
