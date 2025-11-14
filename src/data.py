import numpy as np
import pandas as pd
from pandas import json_normalize

# Define a function to get the full text from a tweet object.
# Tweets can be truncated, storing the full version in 'extended_tweet.full_text'.
def extract_full_text(tweet):
    # Start with the standard 'text' field
    text = tweet['text']
    # Check if the 'extended_tweet.full_text' field exists (is not NaN)
    if not pd.isna(tweet['extended_tweet.full_text']):
        # If it exists, it's the full text, so use it instead
        text = tweet['extended_tweet.full_text']
    return text

def load_train(path="data/raw/train.jsonl"):
    df = pd.read_json(path, lines=True) # Load the training data from a JSON Lines file (one JSON object per line)
    df = json_normalize(df.to_dict(orient="records")) # The tweet data is nested. json_normalize flattens the nested JSON into columns.
    X = df.drop("label", axis=1) # Separate features from the target variable for the training set
    y = df["label"]
    X["full_text"] = X.apply(extract_full_text, axis=1) # Apply this function to every row (axis=1) in the training data
    return X, y

def load_test(path="data/raw/kaggle_test.jsonl"):
    df = pd.read_json(path, lines=True) # Load the Kaggle test data (which we will make predictions on)
    df = json_normalize(df.to_dict(orient="records")) # Also normalize the Kaggle data
    df["full_text"] = df.apply(extract_full_text, axis=1) # Apply the same function to the Kaggle test data
    return df, df["challenge_id"]