# Group_14 - Intro to Machine Learning Project

## 1. Project Overview

This repository contains the full source code, datasets, notebooks, and report for a project with two main parts:

- **Part 1:** Regression
- **Part 2:** Classification

The main goal is to keep the project reproducible, team-friendly, and aligned with the required submission structure.

## 2. Project Structure

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

## 3. Datasets

- **Dataset description:** Read `data/README.md` for more information.
- **Detailed summary:** See `docs/DATASET_SUMMARY.md` for a detailed dataset overview.

## 4. Environment Setup

### 4.1 Python Version (Required)

> WARNING: All team members must use Python 3.11.x to ensure consistent and reproducible results.

- Required version: **Python 3.11.x**
- Do not use Python 3.12+ or 3.10- to avoid dependency and notebook kernel issues.

### 4.2 Create a Virtual Environment

#### venv (Recommended)

```bash
python3.11 -m venv .venv
```

### 4.3 Activate Environment

- **4.3.1 Windows PowerShell**

```powershell
.\.venv\Scripts\Activate.ps1
```

- **4.3.2 Windows Command Prompt**

```bat
.venv\Scripts\activate.bat
```

- **4.3.3 macOS/Linux**

```bash
source .venv/bin/activate
```

### 4.4 Install Dependencies

```bash
pip install -r requirements.txt
```

### 4.5 Register Jupyter Kernel

```bash
python -m ipykernel install --user --name=ml-project-py311
```

Then select the kernel:

```text
ml-project-py311
```

### 4.6 Verify Python Version

```bash
python --version
```

Expected output:

```text
Python 3.11.x
```

### 4.7 Team Rules

- **4.7.1** Do not commit code executed with a different Python version.
- **4.7.2** If you face runtime or package errors, verify Python version first.
- **4.7.3** Notebooks must run end-to-end from a clean kernel without errors.


## 5. Quick Start

### 5.1 Set up and activate environment

PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Command Prompt:

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate.bat
```

### 5.2 Install dependencies

```powershell
pip install -r requirements.txt
```

### 5.3 Open notebooks

Run in the following order:

1. `code/Part1_Regression/notebook.ipynb`
2. `code/Part2_Classification/notebook.ipynb`

## 6. Reproducibility

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

## 7. Internal Documents

- `docs/REQUIREMENT.md`: assignment details and technical requirements.
- `docs/PLAN.md`: team allocation, timeline, and branch naming.
- `docs/DATASET_SUMMARY.md`: summary of the two selected datasets.
- `data/README.md`: raw/processed data management conventions.

## 8. Team Workflow

### 8.1 Branch Naming

According to the current plan, task branches follow this format:

```text
task/<phase>/<member>-<task>
```

Examples:

- `task/1/A-data-description`
- `task/2/B-gradient-descent`
- `task/6/E-final-report`

### 8.2 Commit Convention

Commit format:

```text
type(scope): short description
```

Examples:

- `feat(regression): add ridge regression notebook section`
- `fix(classification): stabilize irls convergence`
- `docs(report): update experiment discussion`

Read more [here](https://gist.github.com/qoomon/5dfcdf8eec66a051ecd85625518cfd13)
