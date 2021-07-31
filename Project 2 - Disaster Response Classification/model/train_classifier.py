# loading sys
import sys
# loading data pipeline related libraries
import pandas as pd
import re
import pickle
from sqlalchemy import create_engine

# loading NLP libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# loading ML libraries
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
# - NLP related
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def load_db(database_filepath):
    """
    input :
        - database file path
    output :
        - X for using in ML pipeline
        - y for using in ML pipeline
        - cat_names for specifying the column names in y
    """

    # retrieve sql db
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    # using sql syntax to transform to df
    df = pd.read_sql('select * from des_res', engine)
    # message
    X = df['message']
    # eliminating other columns
    df.drop(labels=['id', 'message', 'original', 'genre'], axis=1, inplace=True)
    # predicted categories
    y = df
    # cat_names for specifying the column names in y
    cat_names = y.columns

    return X, y, cat_names


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
    text = [w for w in text if w not in stopwords.words("english")]
    # lemmatization
    text = [WordNetLemmatizer().lemmatize(w) for w in text]

    return text


def ml_model():
    """
    output: Cross-validation of random forest pipeline
    """

    # pipeline of ML
    pipeline = Pipeline([
        ('vector', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    # searching hyperparams
    parameters = {
        'clf__estimator__max_depth': [10],
        'clf__estimator__n_estimators': [50]
    }

    # build model
    model = GridSearchCV(pipeline, param_grid=parameters, cv=5, verbose=10)

    return model


def evaluate(model, X_test, y_test, cat_names):
    """
    input:
        - ML model
        - X_test: X testing data
        - y_test: y testing data

    output: prints classification report
    """

    # predict y using X_test
    y_pred = model.predict(X_test)
    # make a classification report to check the out come
    report = classification_report(y_test, y_pred, target_names=cat_names)
    print(report)


def save_model(model, model_filepath):
    # save model
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]

        print('Loading DB data...')
        X, y, category_names = load_db(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        print('Building model...')
        model = ml_model()

        print('Training model...')
        model.fit(X_train, y_train)

        print('Evaluating model...')
        evaluate(model, X_test, y_test, category_names)

        print('Saving model...')
        save_model(model, model_filepath)

        print('Congrats! Trained model saved!')

    else:
        print('Error! Please refer the correct input instructed in the repo')


if __name__ == '__main__':
    main()
