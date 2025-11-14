from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords

def build_pipeline():
    """
    SVM classifier pipeline using LinearSVC (best SVM for text classification).
    """
    
    stop_words = stopwords.words("french")
    
    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            stop_words=stop_words,
            max_df=0.7,
            min_df=3,
            max_features=1000,
            ngram_range=(1, 2),
        )),
        ("clf", LinearSVC(
            random_state=42,
        )),
    ])
    return model