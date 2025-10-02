# TDC-LightGBM-Final 🚀

This repo contains **baseline and tuned LightGBM models** for 27 ADME datasets from [Therapeutics Data Commons (TDC)](https://tdcommons.ai/).  
Both models and metrics are included, along with Docker support for easy reproducibility.

## 📂 Structure
- `notebooks/` → main Jupyter notebook pipeline
- `src/` → training & utility scripts
- `configs/` → dataset mappings
- `results/` → evaluation results (CSV)
- `models/` → trained baseline + tuned pickle files
- `environment.yml` / `requirements.txt` → reproducibility
- `Dockerfile` → build a container

## 🔧 Usage

### Clone
```bash
git clone https://github.com/adigopulasgit/TDC-LightGBM-Final.git
cd TDC-LightGBM-Final
