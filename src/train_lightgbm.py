import os, json, joblib
import pandas as pd
import lightgbm as lgb
from data_utils import smiles_to_morgan, eval_regression, eval_binary

# Paths
SPLIT_DIR = "splits"
MODEL_DIR = "models"
RESULTS_FILE = "results/tdc_lgbm_final_metrics.csv"
os.makedirs(MODEL_DIR, exist_ok=True)

# Load tuned params (saved after Optuna/RandomSearch)
with open("configs/tdc_lgbm_best_params.json", "r") as f:
    best_params = json.load(f)

# Define dataset types
DATASET_TYPE = json.load(open("configs/dataset_map.json"))

results = []

def load_split_from_csv(task, name):
    base = f"{task}_{name}"
    return {
        "train": pd.read_csv(os.path.join(SPLIT_DIR, f"{base}_train.csv")),
        "valid": pd.read_csv(os.path.join(SPLIT_DIR, f"{base}_valid.csv")),
        "test":  pd.read_csv(os.path.join(SPLIT_DIR, f"{base}_test.csv")),
    }

for key, params in best_params.items():
    task, name = key.split(".", 1)
    target_type = DATASET_TYPE.get(key, "regression")

    split = load_split_from_csv(task, name)
    df_trainvalid = pd.concat([split["train"], split["valid"]])
    df_test = split["test"]

    X_trva = smiles_to_morgan(df_trainvalid["Drug"].tolist())
    y_trva = df_trainvalid["Y"].values
    X_test = smiles_to_morgan(df_test["Drug"].tolist())
    y_test = df_test["Y"].values

    # Baseline
    if target_type == "regression":
        model = lgb.LGBMRegressor(random_state=42, n_jobs=-1)
        model.fit(X_trva, y_trva)
        yhat = model.predict(X_test)
        metrics = eval_regression(y_test, yhat)
    else:
        model = lgb.LGBMClassifier(random_state=42, n_jobs=-1)
        model.fit(X_trva, y_trva)
        yscore = model.predict_proba(X_test)[:, 1]
        metrics = eval_binary(y_test, yscore)

    joblib.dump(model, os.path.join(MODEL_DIR, f"{task}_{name}_baseline.pkl"))
    results.append({"Dataset": key, "Type": "Baseline", **metrics})

    # Tuned
    if target_type == "regression":
        model = lgb.LGBMRegressor(random_state=42, n_jobs=-1, **params)
        model.fit(X_trva, y_trva)
        yhat = model.predict(X_test)
        metrics = eval_regression(y_test, yhat)
    else:
        model = lgb.LGBMClassifier(random_state=42, n_jobs=-1, **params)
        model.fit(X_trva, y_trva)
        yscore = model.predict_proba(X_test)[:, 1]
        metrics = eval_binary(y_test, yscore)

    joblib.dump(model, os.path.join(MODEL_DIR, f"{task}_{name}_tuned.pkl"))
    results.append({"Dataset": key, "Type": "Tuned", **metrics})

pd.DataFrame(results).to_csv(RESULTS_FILE, index=False)
print(f"✅ Training done. Saved models to {MODEL_DIR}, metrics to {RESULTS_FILE}")

