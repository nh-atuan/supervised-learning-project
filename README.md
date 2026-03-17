# Group_14 - Intro to Machine Learning Project

This repository contains the group project with two main tasks:

- Part 1: Regression
- Part 2: Classification

The project is organized for reproducibility, teamwork, and submission-ready structure.

## Repository Structure

```text
Group_14/
├── report/
│   ├── report.tex
│   └── report.pdf
├── code/
│   ├── Part1_Regression/
│   │   ├── notebook.ipynb
│   │   └── utils.py
│   └── Part2_Classification/
│       ├── notebook.ipynb
│       └── utils.py
├── data/
│   └── README.md
├── docs/
│   └── PLAN.md
├── logs/
├── outputs/
├── requirements.txt
├── .gitignore
└── README.md
```

## Quick Start

### 1) Create virtual environment

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Windows (Command Prompt):

```bat
python -m venv .venv
.venv\Scripts\activate.bat
```

### 2) Install dependencies

```powershell
pip install -r requirements.txt
```

### 3) Run notebooks

Open and run:

- `code/Part1_Regression/notebook.ipynb`
- `code/Part2_Classification/notebook.ipynb`

## Reproducibility Rules

- Always fix random seed before experiments.
- Keep train/test split and hyperparameters logged.
- Use pinned dependencies in `requirements.txt`.

Example:

```python
from code.Part1_Regression.utils import set_seed as set_seed_reg
from code.Part2_Classification.utils import set_seed as set_seed_cls

set_seed_reg(42)
set_seed_cls(42)
```

## Team Workflow

### Branch Strategy

- `main`: stable branch only
- `dev`: integration branch
- `feature/*`: task branches, for example:
	- `feature/eda`
	- `feature/regression-model`
	- `feature/classification-model`
	- `feature/report`

### Commit Message Convention

Use format:

```text
type(scope): short description
```

Examples:

```text
feat(eda): add correlation matrix
fix(model): correct gradient descent bug
docs(report): update section 2.1 theory
```

## Data Policy

- Do not commit large dataset files.
- Keep dataset source and download instructions in `data/README.md`.

## Notes

- `logs/` is for experiment logs.
- `outputs/` is for figures, tables, and exported artifacts.
