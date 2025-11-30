# src/models/camembert_xgb_meta.py
import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# device selection
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

MODEL_NAME = "camembert-base"


class CamembertEmbedder:
    def __init__(self, model_name=MODEL_NAME, device=device, batch_size=16, max_length=128):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        model_path = "/Data/iuliia.korotkova/DL/french_bert_classifier"

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
        self.model.eval()
        
        if self.device == "cuda":
            self.model.half()

    def embed_texts(self, texts):
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            tokens = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                if self.device == "cuda":
                    # mixed precision for speed
                    with torch.cuda.amp.autocast():
                        outputs = self.model(**tokens, output_hidden_states=True)
                else:
                    outputs = self.model(**tokens, output_hidden_states=True)
                last_hidden = outputs.hidden_states[-1]  # (B, T, D)
                mask = tokens["attention_mask"].unsqueeze(-1)  # (B, T, 1)
                masked_hidden = last_hidden * mask
                sum_hidden = masked_hidden.sum(dim=1)
                lengths = mask.sum(dim=1).clamp(min=1e-9)
                mean_pooled = sum_hidden / lengths
                all_embeddings.append(mean_pooled.cpu().numpy())
        return np.vstack(all_embeddings)
    
class TextMetaTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, embedder=None, scale_embeddings=True, scale_metadata=True,
                 num_columns=None, bool_columns=None, list_columns=None, unuseful_columns=None,
                 categorical_column='source'):
        self.embedder = embedder or CamembertEmbedder()
        self.scale_embeddings = scale_embeddings
        self.scale_metadata = scale_metadata
        self.emb_scaler = None
        self.meta_transformer = None  # replaced meta_scaler
        self.num_columns = num_columns
        self.bool_columns = bool_columns
        self.list_columns = list_columns
        self.unuseful_columns = unuseful_columns
        self.categorical_column = categorical_column  # categorical column name
        
    def extract_features(self, df, num_columns, bool_columns, list_columns, unuseful_columns):
        df = df.copy()
        # numerical data
        df[num_columns] = df[num_columns].fillna(0).replace([np.inf, -np.inf], 0)
        # boolean data
        for column in bool_columns:
            df[column] = df[column].map({True: 1, False: 0})
        # list data
        for col in list_columns:
            df[col] = df[col].apply(lambda x: len(x) if isinstance(x, list) else 0)
        # unuseful data
        df = df.drop(unuseful_columns, axis=1)

        return df

    def fit(self, X, y=None):
        # Ensure cache directory exists
        os.makedirs("cache", exist_ok=True)
        cache_path = "cache/embeddings_meta.npy"

        # 1. Try to load cached embeddings
        if os.path.exists(cache_path):
            print("Loading cached embeddings...")
            emb = np.load(cache_path)

        else:
            print("Cache not found. Computing embeddings...")
            texts = X["full_text"].astype(str).tolist()
            emb = self.embedder.embed_texts(texts)

            # Save to cache
            np.save(cache_path, emb)
            print(f"Saved embeddings to {cache_path}")

        # Apply extract_features
        meta = self.extract_features(X, self.num_columns, self.bool_columns, self.list_columns, self.unuseful_columns)

        if self.scale_embeddings:
            self.emb_scaler = StandardScaler()
            self.emb_scaler.fit(emb)
            emb = self.emb_scaler.transform(emb)

        # ColumnTransformer for numeric + categorical columns
        if meta.shape[1] > 0 and self.scale_metadata:
            self.meta_transformer = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), self.num_columns),
                    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), [self.categorical_column])
                ],
                remainder='drop'
            )
            self.meta_transformer.fit(X)
            meta = self.meta_transformer.transform(X)

        return self

    def transform(self, X):
        texts = X["full_text"].astype(str).tolist()
        emb = self.embedder.embed_texts(texts)
        if self.emb_scaler is not None:
            emb = self.emb_scaler.transform(emb)

        meta = self.extract_features(X, self.num_columns, self.bool_columns, self.list_columns, self.unuseful_columns)
        if self.meta_transformer is not None and meta.shape[1] > 0:
            meta = self.meta_transformer.transform(X)

        combined = np.hstack([emb, meta]).astype(np.float32)
        return combined

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        return self.transform(X)

def build_pipeline(num_columns, bool_columns, list_columns, unuseful_columns, n_estimators=1000, max_depth=9):
    feature_transformer = TextMetaTransformer(
        embedder=CamembertEmbedder(batch_size=32, max_length=128),
        scale_embeddings=True,
        scale_metadata=True,
        num_columns=num_columns,
        bool_columns=bool_columns,
        list_columns=list_columns,
        unuseful_columns = unuseful_columns
    )

    clf = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        n_jobs=4,
        random_state=42          
    )

    return Pipeline([
        ("features", feature_transformer),
        ("clf", clf)
    ])