from nltk.tokenize import word_tokenize
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine
import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

def tokenize_text(input_text):
    tokens = word_tokenize(input_text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for token in tokens:
        clean_token = lemmatizer.lemmatize(token).lower().strip()
        clean_tokens.append(clean_token)

    return clean_tokens

# load model
model = joblib.load("../models/classifier.pkl")

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
data_df = pd.read_sql_table('DisasterResponse_table', engine)

# index webpage displays visuals and receives user input for model
@app.route('/')
@app.route('/index')
def index():
    # extract data for visuals
    genre_counts = data_df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    category_names = data_df.iloc[:, 4:].columns
    category_counts = (data_df.iloc[:, 4:] != 0).sum()
    
    # create visuals
    visualizations = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(visualizations)]
    graphs_json = json.dumps(visualizations, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphs_json=graphs_json)

# web page for user query and displaying model results
@app.route('/go')
def go():
    # save user input as query
    user_query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([user_query])[0]
    classification_results = dict(zip(data_df.columns[4:], classification_labels))

    # Render the go.html file
    return render_template(
        'go.html',
        query=user_query,
        classification_result=classification_results
    )

def main():
    app.run(host='0.0.0.0', port=3001, debug=True)

if __name__ == '__main__':
    main()
