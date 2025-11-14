from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

'''
Create a scikit-learn Pipeline. This chains steps together.
Data will flow from 'tfidf' (text to numbers) to 'clf' (classifier).
'''
def build_pipeline():
    stop_words = stopwords.words("french") # Load a list of common French stop words (e.g., 'le', 'la', 'de')

    model = Pipeline([
        # Step 1: TfidfVectorizer - converts text into a matrix of TF-IDF features
        ("tfidf", TfidfVectorizer(
            stop_words=stop_words, # Remove French stop words
            max_df=0.7, # Ignore words that appear in > 70% of tweets (too common)
            min_df=3, # Ignore words that appear in < 3 tweets (too rare)
            max_features=1000, # Keep only the top 1000 features
            ngram_range=(1, 2),  # Include 1-word (unigrams) and 2-word (bigrams) sequences
        )),
        # Step 2: Classifier - Logistic Regression
        ("clf", LogisticRegression(
            solver="liblinear", # Good solver for this type of problem
            random_state=42, # For reproducible results
        )),
    ])
    return model