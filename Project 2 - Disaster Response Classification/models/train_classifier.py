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
from custom_transformer import DisResExtractor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
# - NLP related
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def load_data(database_filepath):
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
    df = pd.read_sql('select * from dis_res', engine)
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
    text = [word for word in text if word not in stopwords.words("english")]

    # lemmatization
    lemmatizer = WordNetLemmatizer()

    tokenized_text = []
    for word in text:
        tokenized_text.append(lemmatizer.lemmatize(word).lower().strip())

    return tokenized_text


def build_model():
    """
    output: Cross-validation of random forest pipeline
    """

    # pipeline of ML
    pipeline = Pipeline([
        ('features',
         FeatureUnion([
            ('text_pipeline',
             Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                       ('tfidf', TfidfTransformer())])),
            ('verb', DisResExtractor())])),
        ('clf', RandomForestClassifier())
    ])

    # searching hyperparams you can dilate the search range ...
    parameters = {
        'clf__n_estimators': [50],
    }

    # build model
    model = GridSearchCV(pipeline, param_grid=parameters, verbose=10)

    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """
    input:
        - ML model
        - X_test: X testing data
        - y_test: y testing data

    output: prints classification report
    """

    # predict y using X_test
    Y_pred = model.predict(X_test)
    # make a classification report to check the out come
    report = classification_report(Y_test, Y_pred, target_names=category_names)
    print(report)


def save_model(model, model_filepath):
    """
    Saving trained model on on disk to be load when required.
    input:
        model = trained classifier
        model_filepath = file path

    """

    # save model
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database ',
              'as the first argument and the filepath of the pickle file to ',
              'save the model to as the second argument. \n\nExample: python ',
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
