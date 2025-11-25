import os
import pandas as pd
import numpy as np
import importlib
import datetime
import sys
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

from src.data import load_train, load_test

def load_model_builder(model_name):
    module_path = f"src.models.{model_name}"
    module = importlib.import_module(module_path)
    return module.build_pipeline

def train(model_name="logistic_regression", if_kfold = False):
    X, y = load_train()
    X_test, ids = load_test()

    build_pipeline = load_model_builder(model_name)
    
    if if_kfold:

        print("Running 5-Fold Cross-Validation on training data------------------------")
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        scores = []
        for fold_idx, (train_idx, val_idx) in enumerate(tqdm(kfold.split(X, y), total=kfold.get_n_splits(), desc="CV folds")):
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
            
            model_fold = build_pipeline()
            model_fold.fit(X_train_fold, y_train_fold)

            y_pred = model_fold.predict(X_val_fold)
            score = accuracy_score(y_val_fold, y_pred)
            scores.append(score)

            print(f"Fold {fold_idx+1} Accuracy: {score * 100:.2f}%")

        print(f"Mean K-Fold Accuracy: {np.mean(scores) * 100:.2f}%")
        print(f"Std Dev K-Fold Accuracy: {np.std(scores) * 100:.2f}%")
        
    
    print("Training final model on all training data-------------------------------")
    final_model = build_pipeline()
    final_model.fit(X, y)
    print("Training complete.")

    preds = final_model.predict(X_test)

    os.makedirs("outputs", exist_ok=True)
    date_str = datetime.datetime.now().strftime("%Y%m%d")
    output_path = f"outputs/{model_name}_{date_str}.csv"
    pd.DataFrame({"ID": ids, "Prediction": preds}).to_csv(output_path, index=False)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "logistic_regression"
    train(model_name)