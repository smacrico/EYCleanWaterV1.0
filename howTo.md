cd project
python -m venv .venv
# Windows:  .venv\Scripts\activate
source .venv/bin/activate

# Editable install exposes the CLI command `ey-train`
pip install -e .

# Make sure your processed training parquet exists at:
# data/processed/engineered_features.parquet
# (or point to a different file with --train-data)


ey-train --config configs/train_config.yaml --write-submission


# common options
ey-train --help
ey-train --train-data data/processed/my_features.parquet
ey-train --targets drp             # train DRP only
ey-train --no-submission           # skip writing submission CSV

# from prompt creation ####
# Setup environment
cd C:\Users\XP222SP\ey-water-quality-challenge
conda env create -f environment.yml
conda activate ey-water-challenge

# Run notebooks sequentially
jupyter notebook notebooks/01_improved_benchmark_model.ipynb

# Or use CLI for production training
python -m src.cli_train --targets all --generate-submission