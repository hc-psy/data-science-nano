import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
# from plotly.graph_objs import Bar
from sqlalchemy import create_engine

from plotly.graph_objs import Bar


import re
from nltk.corpus import stopwords

# from sklearn.externals import joblib
import joblib

app = Flask(__name__)


def tokenize(text):
    """
    input:
        - text
    output:
        - tokenized text
    """

    # to lower case
    text = text.lower()
    # substitute the foreign letters
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    # tokenization
    text = word_tokenize(text)
    # eliminating stopwords
    text = [word for word in text if word not in stopwords.words("english")]

    # lemmatization
    lemmatizer = WordNetLemmatizer()

    tokenized_text = []
    for word in text:
        tokenized_text.append(lemmatizer.lemmatize(word).lower().strip())

    return tokenized_text


# load data
engine = create_engine('sqlite:///data/DisasterResponse.db')
df = pd.read_sql('select * from dis_res', engine)

# load model
model = joblib.load("models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    columns_count = df.astype(bool).sum(axis=0).iloc[4:] / len(df)
    columns = list(df.astype(bool).sum(axis=0).iloc[4:].index)

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
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
                    x=columns,
                    y=columns_count
                )
            ],

            'layout': {
                'title': 'Frequency table of each category shown by percentage',
                'yaxis': {
                    'title': "Percentage"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
