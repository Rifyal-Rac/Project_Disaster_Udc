import json
import plotly
import pandas as pd

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from flask import Flask
from flask import render_template, request
from plotly.graph_objs import Bar

from sqlalchemy import create_engine
import joblib


app = Flask(__name__)

def tokenize_text(input_text):
    tokens = word_tokenize(input_text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for token in tokens:
        clean_token = lemmatizer.lemmatize(token).lower().strip()
        clean_tokens.append(clean_token)

    return clean_tokens

# model created load
model = joblib.load("./models/classifier.pkl")

# loading the data data
engine = create_engine('sqlite:///./data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse_table', engine)

# webpage visual
@app.route('/')
@app.route('/index')
def index():
    # extract data for visuals
    count_genre = df.groupby('genre').count()['message']
    name_genre = list(count_genre.index)
    
    name_category = df.iloc[:, 4:].columns
    count_category = (df.iloc[:, 4:] != 0).sum()
    
    # create visuals
    visualizations = [
        {
            'data': [
                Bar(
                    x=name_genre,
                    y=count_genre
                )
            ],

            'layout': {
                'title': 'Distribution of Message: Genres ',
                'yaxis': {
                    'title': "Number"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        
        {
            'data': [
                Bar(
                    x=name_category,
                    y=count_category
                )
            ],

            'layout': {
                'title': 'Distribution of Message: Categories',
                'yaxis': {
                    'title': "Number"
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
    classification_results = dict(
        zip(
            df.columns[4:], 
            classification_labels
            )
            )

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
