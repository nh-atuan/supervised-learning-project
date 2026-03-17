import os
import random
from datetime import datetime

import numpy as np


def set_seed(seed: int = 42) -> None:
	"""Set random seed for reproducible experiments."""
	os.environ["PYTHONHASHSEED"] = str(seed)
	random.seed(seed)
	np.random.seed(seed)


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
