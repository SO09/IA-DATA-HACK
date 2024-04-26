from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import nltk
import joblib

from nltk import word_tokenize

def make_model(X_train, y_train):
    model = make_pipeline(CountVectorizer(tokenizer=word_tokenize, ngram_range=(3, 3)), LogisticRegression(max_iter=500, random_state=42))
    model.fit(X_train, y_train)
    return model
    
if __name__ == "__main__":
    
    nltk.download('punkt')
    
    file_path = "data/data.csv"

    df = pd.read_csv(file_path).drop(['src'], axis=1)

    X = df['text']
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

    model = make_model(X_train, y_train)
    
    joblib.dump(model, 'models/logistic_regression_model.pkl')