import json
import plotly
import pandas as pd

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from flask import Flask
from flask import render_template, request
from plotly.graph_objs import Bar
from plotly.utils import PlotlyJSONEncoder

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
    
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    category_names = df.iloc[:, 4:].columns
    category_counts = (df.iloc[:, 4:]!=0).sum()
    genre_category_counts = df.groupby('genre').sum().iloc[:, 1:]
    genre_category_names = list(genre_category_counts.columns)

    genre_visualization = {
        'data': [
            Bar(
                x=genre_category_names,
                y=genre_category_counts.loc[genre].values,
                name=genre
            ) for genre in genre_names
        ],

        'layout': {
            'title': 'Distribution of Message Categories by Genre',
            'barmode': 'group',
            'yaxis': {
                'title': 'Count'
            },
            'xaxis': {
                'title': 'Category'
            }
        }
    }

    category_visualization = {
        'data': [
            Bar(
                x=category_names,
                y=category_counts,
                marker_color='blue',
                name='Not Relevant'
            ),
            Bar(
                x=category_names,
                y=len(df) - category_counts,
                marker_color='orange',
                name='Relevant'
            )
        ],

        'layout': {
            'title': 'Distribution of Message Categories',
            'barmode': 'stack',
            'yaxis': {
                'title': 'Count'
            },
            'xaxis': {
                'title': 'Category'
            }
        }
    }
    visualizations = [
        genre_visualization,
        category_visualization
        ]

    
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(visualizations)]
    graphs_json = json.dumps(visualizations, cls=PlotlyJSONEncoder)
    
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
