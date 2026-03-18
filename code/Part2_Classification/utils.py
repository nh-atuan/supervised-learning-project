import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def set_seed(seed: int = 42) -> None:
	"""Set random seed for reproducible experiments."""
	os.environ["PYTHONHASHSEED"] = str(seed)
	random.seed(seed)
	np.random.seed(seed)


def load_dataset(path: str) -> pd.DataFrame:
	"""Load a CSV dataset from disk."""
	path_obj = Path(path)
	if not path_obj.exists():
		raise FileNotFoundError(f"Dataset not found: {path}")
	if path_obj.suffix.lower() != ".csv":
		raise ValueError("Only CSV files are supported")
	return pd.read_csv(path_obj)


def describe_dataset(df: pd.DataFrame) -> dict[str, Any]:
	"""Return a compact summary useful for quick EDA checks."""
	summary = {
		"shape": df.shape,
		"columns": list(df.columns),
		"dtypes": df.dtypes.astype(str).to_dict(),
		"missing_per_column": df.isna().sum().to_dict(),
		"numeric_description": df.describe(include=[np.number]).T.to_dict(),
	}
	return summary


def train_val_test_split(
	X,
	y,
	train_ratio: float = 0.7,
	val_ratio: float = 0.1,
	seed: int = 42,
):
	"""Split arrays/dataframes into train/validation/test sets reproducibly."""
	if not (0 < train_ratio < 1):
		raise ValueError("train_ratio must be in (0, 1)")
	if not (0 <= val_ratio < 1):
		raise ValueError("val_ratio must be in [0, 1)")
	if train_ratio + val_ratio >= 1:
		raise ValueError("train_ratio + val_ratio must be < 1")
	if len(X) != len(y):
		raise ValueError("X and y must have the same number of samples")

	n_samples = len(X)
	rng = np.random.default_rng(seed)
	indices = np.arange(n_samples)
	rng.shuffle(indices)

	train_end = int(train_ratio * n_samples)
	val_end = train_end + int(val_ratio * n_samples)

	idx_train = indices[:train_end]
	idx_val = indices[train_end:val_end]
	idx_test = indices[val_end:]

	def _slice(obj, idx):
		if hasattr(obj, "iloc"):
			return obj.iloc[idx]
		return obj[idx]

	return (
		_slice(X, idx_train),
		_slice(X, idx_val),
		_slice(X, idx_test),
		_slice(y, idx_train),
		_slice(y, idx_val),
		_slice(y, idx_test),
	)


def train_test_indices(n_samples: int, test_size: float = 0.2, seed: int = 42):
	"""Return reproducible shuffled train/test indices."""
	if not 0 < test_size < 1:
		raise ValueError("test_size must be in (0, 1)")
	rng = np.random.default_rng(seed)
	indices = np.arange(n_samples)
	rng.shuffle(indices)
	split = int(n_samples * (1 - test_size))
	return indices[:split], indices[split:]


def log_experiment_params(params: dict, prefix: str = "[Classification]") -> None:
	"""Print experiment parameters with a timestamp for tracking."""
	timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
	print(f"{prefix} {timestamp}")
	for key, value in params.items():
		print(f"- {key}: {value}")
