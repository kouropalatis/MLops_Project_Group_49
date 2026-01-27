# MLOps: Financial News Sentiment Analysis

An end-to-end MLOps pipeline designed to **quantify market sentiment** from financial news articles.
This project demonstrates a production-grade workflow for an NLP model that classifies financial texts as *Positive*, *Negative*, or *Neutral*, providing actionable signals for **algorithmic trading** and market analysis.

## Features

- ✅ Modular and scalable project structure
- ✅ Reproducible environments using uv
- ✅ Dockerized training and inference
- ✅ CI/CD with GitHub Actions
- ✅ Unit testing
- ✅ Clear data separation (raw vs processed)
- ✅ Model versioning
- ✅ Experiment reproducibility

## Project Structure

```
.
├── .devcontainer/            # Dev container configuration
├── .github/
│   └── workflows/
│       └── tests.yaml        # CI pipeline
├── configs/                  # Configuration files
├── data/
│   ├── raw/                  # Raw datasets
│   └── processed/            # Processed datasets
├── dockerfiles/
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/
│   ├── mkdocs.yaml
│   └── source/
│       └── index.md
├── models/                   # Trained model artifacts
├── notebooks/                # Jupyter notebooks
├── reports/
│   └── figures/              # Generated plots
├── src/
│   └── project/
│       ├── __init__.py
│       ├── backend.py
│       ├── data.py
│       ├── evaluate.py
│       ├── model.py
│       ├── train.py
│       └── visualize.py
├── tests/
│   ├── __init__.py
│   ├── test_backend.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml
├── README.md
├── requirements.txt
├── requirements_dev.txt
└── tasks.py
```

## Getting Started

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/konstandinoseng/MLops_Project_Group_49)


### Clone the Repository

```bash
git clone https://github.com/konstandinoseng/MLops_Project_Group_49.git
cd your-repo-name
```

## Dependency Management (Using uv)

This project uses `uv` instead of pip for faster dependency management.

### Install uv

**macOS / Linux:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**

```powershell
irm https://astral.sh/uv/install.ps1 | iex
```

### Create Virtual Environment

```bash
uv venv
```

**Activate it:**

**macOS / Linux:**
```bash
source .venv/bin/activate
```

**Windows:**
```bash
.venv\Scripts\activate
```

### Install Dependencies

```bash
uv pip install -r requirements.txt
```

For development dependencies:

```bash
uv pip install -r requirements_dev.txt
```

## Running the Project

### Training

```bash
python src/project/train.py
```

### Evaluation

```bash
python src/project/evaluate.py
```

### API

```bash
uv run fastapi dev src/project/backend.py
```

### Frontend (User Interface)

Open a new terminal and run:

```bash
uv run streamlit run src/project/frontend.py
```

```bash
uv run streamlit run src/project/frontend.py
```

### 💡 Usage Guide

Once the Frontend is running (usually at `http://localhost:8501`):
1.  **Open the link** in your browser.
2.  **Paste a URL** of a financial news article (e.g., from *Bloomberg*, *Reuters*, or *Yahoo Finance*).
3.  Click **"Analyze"**.
4.  The system will output the **Overall Sentiment** (Positive/Neutral/Negative) and highlight specific sentences that influenced the decision.

## Docker

### Build Training Image

```bash
docker build -f dockerfiles/train.dockerfile -t project-train .
```

### Build API Image

```bash
docker build -f dockerfiles/backend.dockerfile -t project-backend .
```

## Testing

Run all tests:

```bash
pytest
```

## CI/CD

This project uses GitHub Actions for:

- Linting
- Formatting
- Unit tests
- Build checks

Workflow file: `.github/workflows/tests.yaml`

## Data Management

- **Raw data:** `data/raw/`
- **Processed data:** `data/processed/`
- **Models:** `models/`
- **Reports:** `reports/`

These directories should typically not be tracked by Git.

## Pre-commit Hooks

### Install Hooks

```bash
pre-commit install
```

### Run Manually

```bash
pre-commit run --all-files
```

## Documentation

Serve documentation locally:

```bash
mkdocs serve
```

## License

This project is licensed under the MIT License. See LICENSE for details.

---
