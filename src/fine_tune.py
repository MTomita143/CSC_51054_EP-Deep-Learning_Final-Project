import json
import numpy as np
import pandas as pd
from pandas import json_normalize
from tqdm import tqdm
import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, AutoModel
from transformers import get_linear_schedule_with_warmup  # scheduler
from torch.optim import AdamW
from sklearn.preprocessing import StandardScaler
from src.data import load_train, load_test

# device selection
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

MODEL_NAME = "camembert-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
bert_model = AutoModel.from_pretrained(MODEL_NAME).to(device)

class TweetDataset(Dataset):
    def __init__(self, texts, metadata, labels=None, max_len=128):
        self.texts = texts
        self.metadata = torch.tensor(metadata, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long) if labels is not None else None
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        tokens = tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        input_ids = tokens['input_ids'].squeeze(0)
        attention_mask = tokens['attention_mask'].squeeze(0)
        
        item = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'metadata': self.metadata[idx]
        }
        if self.labels is not None:
            item['labels'] = self.labels[idx]
        return item
    
    
class CamemBERTClassifier(nn.Module):
    def __init__(self, bert_model, metadata_dim, num_classes):
        super().__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Sequential(
            nn.Linear(bert_model.config.hidden_size + metadata_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, input_ids, attention_mask, metadata):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = outputs.last_hidden_state[:,0,:]  # CLS token
        x = torch.cat([cls_emb, metadata], dim=1)
        x = self.dropout(x)
        logits = self.classifier(x)
        return logits
    
def freeze_bert_layers(model, num_unfrozen_last_layers=7):
    total_layers = model.bert.config.num_hidden_layers
    
    for name, param in model.named_parameters():
        
        # We only care about freezing the encoder layers
        if "encoder.layer" in name:
            # Use Regex to find the pattern "layer.<number>"
            match = re.search(r"encoder\.layer\.(\d+)", name)
            
            if match:
                layer_idx = int(match.group(1))
                
                if layer_idx < total_layers - num_unfrozen_last_layers:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
        
        # Always keep the classifier head and embeddings trainable
        else:
            param.requires_grad = True

    print(f"Freezing complete. Last {num_unfrozen_last_layers} layers are unfrozen.")
    
def train_model(model, train_dataset, val_dataset, device, epochs=10, batch_size=32, lr=2e-5):
    # Create DataLoaders from the Dataset objects
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    total_steps = len(train_loader) * epochs
    
    # Warmup for 6% of total steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(0.06 * total_steps), 
        num_training_steps=total_steps
    )
    
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    
    best_val_acc = 0.0

    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        correct = 0
        total = 0
        running_loss = 0.0
        
        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            metadata = batch['metadata'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(input_ids=input_ids, attention_mask=attention_mask, metadata=metadata)
            
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            running_loss += loss.item()
            
        train_acc = correct / total
        avg_loss = running_loss / len(train_loader)

        # --- Validation Phase ---
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                metadata = batch['metadata'].to(device)
                labels = batch['labels'].to(device)
                
                logits = model(input_ids=input_ids, attention_mask=attention_mask, metadata=metadata)
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        val_acc = val_correct / val_total
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_camembert_meta.pt")
            
    print("Training complete. Best val acc:", best_val_acc)
    
    
def extract_features(df, num_columns, bool_columns, list_columns, unuseful_columns):
    df = df.copy()
    
    # Numerical columns: fill NAs and replace inf
    df[num_columns] = df[num_columns].fillna(0).replace([np.inf, -np.inf], 0)
    
    # Boolean columns: convert True/False -> 1/0
    for column in bool_columns:
        df[column] = df[column].map({True: 1, False: 0})
    
    # List columns: encode as length of the list
    for col in list_columns:
        df[col] = df[col].apply(lambda x: len(x) if isinstance(x, list) else 0)
    
    # Drop unuseful columns
    df = df.drop(unuseful_columns, axis=1, errors='ignore')
    
    return df
    
# --------------------------
# Initialize and train model
# --------------------------
X, y = load_train()
X_test, ids = load_test()
    
#from Julie's code
X = X.dropna(how='all', axis="columns")
X = X.drop(X.columns.difference(X_test.columns).to_list(), axis=1)

num_columns = X.select_dtypes(include=[np.number]).columns.tolist()
bool_columns = X.select_dtypes(include=[np.bool]).columns.tolist()
list_columns = []
for col in X.columns:
    if X[col].apply(lambda x: isinstance(x, list)).any():
        list_columns.append(col)

unuseful_columns = ["lang", "text", "extended_tweet.full_text", "user.description",
    'retweet_count','favorite_count','quote_count','reply_count','retweeted',
    'favorited','user.default_profile_image','user.protected','user.contributors_enabled'
    ]

num_columns = [col for col in num_columns if col not in unuseful_columns]
bool_columns = [col for col in bool_columns if col not in unuseful_columns]
list_columns = [col for col in list_columns if col not in unuseful_columns]

X = extract_features(X, num_columns, bool_columns, list_columns, unuseful_columns)
X_test = extract_features(X_test, num_columns, bool_columns, list_columns, unuseful_columns)

# Keep only numeric columns shared between train/test
common_numeric_cols = X.select_dtypes(include=[np.number]).columns.intersection(
    X_test.select_dtypes(include=[np.number]).columns
)

metadata_train = X[common_numeric_cols].values.astype(np.float32)
metadata_test = X_test[common_numeric_cols].values.astype(np.float32)

# Standardize metadata
scaler = StandardScaler()
metadata_train = scaler.fit_transform(metadata_train)
metadata_test = scaler.transform(metadata_test)
meta_dim = metadata_train.shape[1]
num_classes = len(np.unique(ids))

train_texts, val_texts, train_meta, val_meta, train_labels, val_labels = train_test_split(
    X['full_text'].tolist(),
    metadata_train,
    y.values,
    test_size=0.1,       # 20% for validation
    random_state=42,
    stratify=y.values
)
train_dataset = TweetDataset(train_texts, train_meta, train_labels)
val_dataset = TweetDataset(val_texts, val_meta, val_labels)

model = CamemBERTClassifier(bert_model, metadata_dim=meta_dim, num_classes=num_classes)

# Freeze layers
freeze_bert_layers(model, num_unfrozen_last_layers=7)

train_model(
    model, 
    train_dataset=train_dataset, 
    val_dataset=val_dataset, 
    device=device, 
    epochs=10, 
    batch_size=32
)