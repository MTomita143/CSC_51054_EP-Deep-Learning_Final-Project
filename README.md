# Project Overview

## Dataset Setup

Download the dataset from the following link: [**Kaggle Competition Dataset**](https://www.kaggle.com/competitions/influencers-or-observers-predicting-social-roles/data)


Place the downloaded files into:
```
data
└── raw
    ├── train.jsonl
    └── kaggle_test.jsonl
```

Note: The `data/` folder is ignored by Git, so each team member must download the dataset manually.

---

## Running the Project

You can train and evaluate models in two ways:

### **1. Using the Notebook**

Open: notebooks/01_baseline.ipynb

### **2. Using the Command Line**
Create your model under: `src/models/` \
Then run any model by specifying its filename (without .py):


```bash
python3 -m src.train logistic_regression
```

---

## Output (command line)

Predictions are saved in:
```
outputs/<model_name>_<YYYYMMDD>.csv
```

---

## Result (to be updated)

| Date  | Model               | Mean K-Fold Accuracy (%) |
| ----- | ------------------- | ------------------------ |
| 14/11 | Logistic Regression | 62.59                    |
| 14/11 | SVM                 | 62.55                    |
| 16/11	|Camembert-base	|70.99|
| 20/11 | Logistic Regression with metadata | 75.63                    |
| 20/11 | XGBoost Classification | 77.35                    |
| 25/11 | XGBoost on CamemBERT embeddings + metadata with PCA | 82.52                    |
| 25/11 | n_est = 1000, depth = 9, pca = 128 | 83.31                    |


