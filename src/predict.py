import argparse
import os, joblib
import numpy as np
from data_utils import smiles_to_morgan

MODEL_DIR = "models"

def load_model(dataset, mode):
    path = os.path.join(MODEL_DIR, f"{dataset}_{mode}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)

def main():
    parser = argparse.ArgumentParser(description="Run LightGBM prediction on a SMILES input.")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name, e.g., ADME.Caco2_Wang")
    parser.add_argument("--smiles", type=str, required=True,
                        help="Input SMILES string to predict")
    parser.add_argument("--mode", type=str, default="tuned",
                        choices=["baseline", "tuned"], help="Model type")
    args = parser.parse_args()

    model = load_model(args.dataset, args.mode)

    X = smiles_to_morgan([args.smiles])
    yhat = model.predict(X)

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)[:, 1][0]
        print(f"✅ Predicted probability = {prob:.4f} for {args.smiles}")
    else:
        print(f"✅ Predicted value = {yhat[0]:.4f} for {args.smiles}")

if __name__ == "__main__":
    main()
