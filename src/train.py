import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
import importlib
import datetime
import sys

from src.data import load_train, load_test
from src.models.logistic_regression import build_pipeline #CHANGE MODEL HERE


def load_model_builder(model_name):
    """
    Import src.models.<model_name>.build_pipeline
    """
    module_path = f"src.models.{model_name}"
    module = importlib.import_module(module_path)
    return module.build_pipeline 

def train(model_name="logistic_regression"):
    X, y = load_train()
    X_test, ids = load_test()
    
    build_pipeline = load_model_builder(model_name)
    model = build_pipeline()

    print("Running 5-Fold Cross-Validation on training data------------------------")
    '''
    Use StratifiedKFold to ensure class proportions are maintained in each fold
    This is important for datasets that might be imbalanced
    '''
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    '''
    cross_val_score will train and test the pipeline 5 times
    using the K-fold splits of the *training data*
    '''
    scores = cross_val_score(model, X["full_text"], y, cv=kfold, scoring="accuracy")
    print(f"Mean K-Fold Accuracy: {np.mean(scores) * 100:.2f}%")
    print(f"Std Dev K-Fold Accuracy: {np.std(scores) * 100:.2f}%")

    print("Training final model on all training data-------------------------------")
    model.fit(X["full_text"], y)
    print("Training complete.")

    preds = model.predict(X_test["full_text"])

    '''
    Prepare the submission file
    Combine the 'challenge_id' from the Kaggle data with our predictions
    '''
    os.makedirs("outputs", exist_ok=True)
    
    date_str = datetime.datetime.now().strftime("%Y%m%d")
    output_path = f"outputs/{model_name}_{date_str}.csv"
    df = pd.DataFrame({"ID": ids, "Prediction": preds})
    df.to_csv(output_path, index=False)
    
    print(f"Saved {output_path}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    else:
        model_name = "logistic_regression"  # default

    train(model_name)