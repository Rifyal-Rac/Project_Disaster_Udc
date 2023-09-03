import sys
import pandas as pd
import os
import re
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sqlalchemy import create_engine

def load_data(input_filepath):
    """
    INPUT:
    input_filepath - Filepath to SQLite database
    
    OUTPUT:
    X - Messages (input variable)
    y - Categories of the messages (output variable)
    category_names - Category names for y
    """
    engine = create_engine('sqlite:///' + input_filepath)
    data_df = pd.read_sql_table('DisasterResponse_table', engine)
    
    X = data_df['message']
    y = data_df.iloc[:, 4:]
    category_names = y.columns
    return X, y, category_names


def clean_text(text):
    """
    INPUT:
    text - raw text
    
    OUTPUT:
    cleaned_text - cleaned text after removing URLs and special characters
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    cleaned_text = re.sub(r'[^a-zA-Z0-9]', ' ', text.lower())
    return cleaned_text

def tokenize_text(text):
    """
    INPUT:
    text - cleaned text
    
    OUTPUT:
    tokens - tokenized text
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).strip()
        clean_tokens.append(clean_tok)
        
    return clean_tokens

def train_model(X_train, Y_train, category_names, classifier=AdaBoostClassifier()):
    """
    INPUT:
    X_train - training messages
    Y_train - categories for training messages
    category_names - category names for Y
    classifier - ML classifier (default: AdaBoostClassifier)
    
    OUTPUT:
    model - trained ML model
    """
    pipeline = Pipeline([
        ('text_vec', CountVectorizer(tokenizer=tokenize_text)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(classifier))
    ])
    
    parameters = {
        'clf__estimator__n_estimators': [50, 100],
        'clf__estimator__learning_rate': [0.1, 0.5]
    }
    
    model = GridSearchCV(pipeline, param_grid=parameters, verbose=3, n_jobs=-1, cv=3)
    model.fit(X_train, Y_train)
    
    return model

def evaluate_model(model, X_test, Y_test, category_names):
    """
    INPUT:
    model - trained ML model
    X_test - test messages
    Y_test - categories for test messages
    category_names - category names for Y
    
    OUTPUT:
    None - print classification report
    """
    Y_pred = model.predict(X_test)
    print(classification_report(Y_test, Y_pred, target_names=category_names))

def save_model(model, model_file):
    """
    INPUT:
    model - trained ML model
    model_file - path to save the model
    
    OUTPUT:
    None - save the model to the specified path
    """
    with open(model_file, 'wb') as file:
        pickle.dump(model, file)

def main():
    if len(sys.argv) == 4:
        data_file, cats_file, model_file = sys.argv[1:]
        
        print('Loading data...')
        messages, categories = load_data(data_file, cats_file)
        X = messages['message']
        Y = categories.iloc[:, 4:]
        category_names = Y.columns
        
        print('Cleaning text...')
        X = X.apply(clean_text)
        
        print('Tokenizing text...')
        X = X.apply(tokenize_text)
        
        print('Splitting data...')
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Training model...')
        model = train_model(X_train, Y_train, category_names)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)
        
        print('Saving model...')
        save_model(model, model_file)
        
        print('Model trained and saved!')
    
    else:
        print('Usage: python train_classifier.py <data_file> <categories_file> <model_file>')

if __name__ == '__main__':
    main()