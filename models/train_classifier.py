# import libraries
import pandas as pd
import sys
import pickle

from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline

def load_data(database_filepath):
    '''
    Load the data from DB

    Input : 
    database_filepath - target DB

    Output :
    X,Y, category names - feeding data training
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('DisasterResponse_table', con=engine)
    X = df['message']
    y = df.iloc[:, 4:]
    category_names = y.columns
    return X, y, category_names


def tokenize_text(text):
    '''
    tonkenize_text
    '''
    
    # Import and load resources inside the function
    import re
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer

    # Ensure the resources are available in each spawned process
    nltk.download(['stopwords','punkt', 'wordnet','averaged_perceptron_tagger'], quiet=True)
    # Normalize text: lower case and remove punctuations
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    # Tokenize text
    tokens = word_tokenize(text)
    # Remove stop words
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatization: Reduce words to their base form
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(word).strip() for word in tokens]

    return clean_tokens    

def build_model():
    '''
    Build pipeline model using AdaBoostClassifier  and Grid Search CV
    '''
    pipeline = Pipeline([
        ('text_vec', CountVectorizer(tokenizer=tokenize_text)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])

    parameters = {
        'clf__estimator__n_estimators': [10, 30],
        'clf__estimator__learning_rate': [0.1, 0.5]
    }

    model = GridSearchCV(pipeline, param_grid=parameters, verbose=3, n_jobs=-1, cv=3)
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    print(classification_report(Y_test, Y_pred, target_names=category_names))


def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1], sys.argv[2]

        print('Loading data...')
        X, y, category_names = load_data(database_filepath)

        print('Splitting data...')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        print('Saving model...')
        save_model(model, model_filepath)

        print('Model trained and saved!')

    else:
        print('Usage: python train_classifier.py <database_filepath> <model_filepath>')

if __name__ == '__main__':
    main()
