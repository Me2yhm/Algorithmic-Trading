

from pathlib import Path

from AlgorithmicStrategy import (
    Standarder
)

raw_data_folder = Path.cwd() / "DATA/ML/RAW"
norm_data_folder = Path.cwd() / "DATA/ML/NORM"
label_data_folder = Path.cwd() / "DATA/ML/LABEL"

if not norm_data_folder.exists():
    norm_data_folder.mkdir(parents=True, exist_ok=True)

standard = Standarder(file_folder=raw_data_folder, train=True)
standard.fresh_files()
standard.read_files()
standard.fit_transform(output=norm_data_folder)

