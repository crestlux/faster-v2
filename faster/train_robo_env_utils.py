from pathlib import Path

_ROBOMIMIC_DATASET_DIRS = ("mh", "ph")
_ROBOMIMIC_USE_RAW=("raw", "low_dim", "image")

def _resolve_robomimic_dataset_path(robomimic_root, env_name, dataset_dir, use_image_obs=False):
    assert dataset_dir in _ROBOMIMIC_DATASET_DIRS, f"Expected dataset_dir in {_ROBOMIMIC_DATASET_DIRS}, got {dataset_dir!r}"
    robomimic_root = Path(robomimic_root)
    fname = "image_v141.hdf5" if use_image_obs else "low_dim_v141.hdf5"
    modality = "image" if use_image_obs else "low_dim"

    candidates = [
        robomimic_root / modality / env_name / dataset_dir / fname,
        robomimic_root / env_name / dataset_dir / fname,
        robomimic_root / f"{env_name}_{dataset_dir}" / fname,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    assert False, (
        f"Could not find robomimic dataset for env_name={env_name!r}, dataset_dir={dataset_dir!r}. Tried: {[str(p) for p in candidates]}"
    )
