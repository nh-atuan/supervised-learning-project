# Group_14 - Intro to Machine Learning Project

This repository contains the full source code, datasets, notebooks, and report for a project with two main parts:

- Part 1: Regression
- Part 2: Classification

The main goal is to keep the project reproducible, team-friendly, and aligned with the required submission structure.

## Current Structure

```text
Group_14/
├── code/
│   ├── Part1_Regression/
│   │   ├── notebook.ipynb
│   │   └── utils.py
│   └── Part2_Classification/
│       ├── notebook.ipynb
│       └── utils.py
├── data/
│   ├── raw/
│   │   ├── regression/
│   │   │   ├── day.csv
│   │   │   └── hour.csv
│   │   └── classification/
│   │       └── covtype.csv
│   ├── processed/
│   │   ├── regression/
│   │   └── classification/
│   └── README.md
├── docs/
│   ├── REQUIREMENT.md
│   ├── PLAN.md
│   └── DATASET_SUMMARY.md
├── logs/
├── outputs/
├── report/
│   ├── report.tex
│   └── report.pdf
├── requirements.txt
└── README.md
```

## Datasets in Use

Read `data/README.md` for more information.

See `docs/DATASET_SUMMARY.md` for a detailed dataset overview.

## Quick Start (Windows)

### 1) Create a virtual environment

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Command Prompt:

```bat
python -m venv .venv
.venv\Scripts\activate.bat
```

### 2) Install dependencies

```powershell
pip install -r requirements.txt
```

### 3) Open notebooks

Run in the following order:

1. `code/Part1_Regression/notebook.ipynb`
2. `code/Part2_Classification/notebook.ipynb`

## Reproducibility

- Always fix random seeds before training.
- Log experiment settings and results in `logs/`.
- Store figures/tables/models in `outputs/`.
- Use pinned package versions in `requirements.txt`.

Seed example:

```python
from code.Part1_Regression.utils import set_seed as set_seed_reg
from code.Part2_Classification.utils import set_seed as set_seed_cls

set_seed_reg(42)
set_seed_cls(42)
```

## Internal Documents

- `docs/REQUIREMENT.md`: assignment details and technical requirements.
- `docs/PLAN.md`: team allocation, timeline, and branch naming.
- `docs/DATASET_SUMMARY.md`: summary of the two selected datasets.
- `data/README.md`: raw/processed data management conventions.

## Team Workflow Conventions

According to the current plan, task branches follow this format:

```text
task/<phase>/<member>-<task>
```

Examples:

- `task/1/A-data-description`
- `task/2/B-gradient-descent`
- `task/6/E-final-report`

Commit format:

```text
type(scope): short description
```

Examples:

- `feat(regression): add ridge regression notebook section`
- `fix(classification): stabilize irls convergence`
- `docs(report): update experiment discussion`

## Notes

- The `logs/` and `outputs/` folders are currently empty and reserved for experiment artifacts.
- The `__pycache__/` and `.ipynb_checkpoints/` folders are environment-generated auxiliary files.
