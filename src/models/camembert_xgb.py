# src/models/camembert_xgb.py
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from xgboost import XGBClassifier

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

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

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
                outputs = self.model(**tokens)
                last_hidden = outputs.last_hidden_state  # (B, T, D)
                mask = tokens["attention_mask"].unsqueeze(-1)  # (B, T, 1)
                masked_hidden = last_hidden * mask
                sum_hidden = masked_hidden.sum(dim=1)
                lengths = mask.sum(dim=1).clamp(min=1e-9)
                mean_pooled = sum_hidden / lengths
                all_embeddings.append(mean_pooled.cpu().numpy())
        return np.vstack(all_embeddings)


class TextMetaTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, embedder=None, scale_embeddings=True, scale_metadata=True, pca_components=128):
        self.embedder = embedder or CamembertEmbedder()
        self.scale_embeddings = scale_embeddings
        self.scale_metadata = scale_metadata
        self.pca_components = pca_components

        self.metadata_cols = None
        self.emb_scaler = None
        self.meta_scaler = None
        self.pca = None

    def fit(self, X, y=None):
        texts = X["full_text"].astype(str).tolist()
        emb = self.embedder.embed_texts(texts)

        # select numeric columns excluding full_text
        self.metadata_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if "full_text" in self.metadata_cols:
            self.metadata_cols.remove("full_text")
        meta = X[self.metadata_cols].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)

        if self.scale_embeddings:
            self.emb_scaler = StandardScaler()
            self.emb_scaler.fit(emb)
            emb = self.emb_scaler.transform(emb)

        if self.scale_metadata and meta.shape[1] > 0:
            self.meta_scaler = StandardScaler()
            self.meta_scaler.fit(meta)
            meta = self.meta_scaler.transform(meta)

        if self.pca_components is not None:
            self.pca = PCA(n_components=self.pca_components, random_state=42)
            self.pca.fit(emb)

        return self

    def transform(self, X):
        texts = X["full_text"].astype(str).tolist()
        emb = self.embedder.embed_texts(texts)
        if self.emb_scaler is not None:
            emb = self.emb_scaler.transform(emb)

        meta = X[self.metadata_cols].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
        if self.meta_scaler is not None and meta.shape[1] > 0:
            meta = self.meta_scaler.transform(meta)

        if self.pca is not None:
            emb = self.pca.transform(emb)

        combined = np.hstack([emb, meta]).astype(np.float32)
        return combined

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        return self.transform(X)


def build_pipeline():
    feature_transformer = TextMetaTransformer(
        embedder=CamembertEmbedder(batch_size=32, max_length=128),
        scale_embeddings=True,
        scale_metadata=True,
        pca_components=128
    )

    clf = XGBClassifier(
        n_estimators=300,
        max_depth=5,
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