import os
import pandas as pd
import numpy as np
import importlib
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from src.data import load_train, load_test

def load_model_builder(model_name):
    module = importlib.import_module(f"src.models.{model_name}")
    return module.build_pipeline

def train_hparams(model_name="camembert_xgb"):
    X, y = load_train()
    build_pipeline = load_model_builder(model_name)

    n_estimators_list = [1000]
    max_depth_list = [9]
    pca_components_list = [32, 64, 128]

    # simple 80/20 split to save time
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    results = []

    print("Hyperparameter tuning...")
    for n_est in n_estimators_list:
        for depth in max_depth_list:
            print(f"\nTesting n_estimators={n_est}, max_depth={depth}")
            for pca_comp in pca_components_list:
                print(f"\nTesting pca_components={pca_comp}")

                model = build_pipeline(
                    n_estimators=n_est,
                    max_depth=depth,
                    pca_components=pca_comp
                )

                model.fit(X_train, y_train)
                preds = model.predict(X_val)
                acc = accuracy_score(y_val, preds)

                print(f"Accuracy: {acc:.4f}")
                results.append((n_est, depth, acc))

    # best hyperparams
    best = max(results, key=lambda x: x[3])
    print("\n==============================")
    print(f"Best params: n_estimators={best[0]}, max_depth={best[1]}, pca_components={best[2]}" )
    print(f"Val accuracy = {best[3]:.4f}")
    print("==============================")

    return best


if __name__ == "__main__":
    train_hparams()