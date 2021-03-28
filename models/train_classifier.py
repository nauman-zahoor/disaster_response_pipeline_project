# import libraries
import sys
import pandas as pd
import numpy as np
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sqlalchemy import create_engine
import os
import nltk
from sklearn.metrics import classification_report

from sklearn.model_selection import ParameterGrid

#from sklearn.externals import joblib
import joblib
import pickle
   


# download
nltk.download(['punkt', 'wordnet'])



def load_data(database_filepath):
    ''' Function that reads data from database and returns X, y and categoriy names
    INPUT:
    database_filepath: string containing path to database
    RETURNS:
    X: numpy array of input values
    y: numpy array of output values
    category_names: list of categories
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql("SELECT * FROM data", engine)

    X = df.loc[:,'message'].values

    category_names = ['related', 'request', 'offer',
           'aid_related', 'medical_help', 'medical_products', 'search_and_rescue',
           'security', 'military', 'child_alone', 'water', 'food', 'shelter',
           'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid',
           'infrastructure_related', 'transport', 'buildings', 'electricity',
           'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
           'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
           'other_weather', 'direct_report']

    Y = df.loc[:,category_names].values

    return X, Y, category_names
    

url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
def tokenize(text):
    ''' Function that cleans text by first removing urls and then tokenizes, lemmatizes and lowers each token.
    INPUT:
    text: string containing input text
    RETURNS:
    clean_tokens: list of tokens

    '''
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''Function that builds a pipeline, defines grisearcg parameters and then retuns gridsearch model
    INPUT:
        None
    Output:
    cv: Gridsearch model

    '''
    pipeline = Pipeline([ ('CountVectorizer',CountVectorizer(tokenizer=tokenize)),
                         ('TfidfTransformer',TfidfTransformer()),
                         ('clf',MultiOutputClassifier(RandomForestClassifier()) )
                    ])

    # Uncoment below parameters to serch more parameters. Just note that it will increase training time
    parameters = {
        #'CountVectorizer__ngram_range': ((1, 1), (1, 2)),
        #'CountVectorizer__max_df': (0.5, 0.75, 1.0),
        #'clf__estimator__max_features': (None, 5000, 10000),
        #'TfidfTransformer__use_idf': (True, False),
        'clf__estimator__n_estimators': [ 100, 200,300],
        'clf__estimator__min_samples_split': [2, 3, 4],
        'clf__estimator__n_jobs':[-1],
    }
    cv = GridSearchCV(pipeline, param_grid=parameters,verbose=10)
    #lets see how many grid points are there
    grid = ParameterGrid(parameters)
    print (f"The total number of parameters-combinations is: {len(grid)}")
    return cv

    


def evaluate_model(model, X_test, Y_test, category_names):
    '''Function that evaluateds the perfoemance of traineid model
    INPUT:
    model: model object
    X_text: test data inputs
    Y_test: test data outputs
    category_names: likst of categories
    RETURNS:
    None
    '''
    y_pred = model.predict(X_test)
    for i,col in enumerate(category_names):
        print('#####################   ',col,'   #####################   ' )
        print(classification_report(Y_test[:,i], y_pred[:,i]))
    print("\nBest Parameters:", model.best_params_)
              


def save_model(model, model_filepath):
    '''Save a model to a pickle file
    INPUT:
    model: model object
    model_filepath: model filename and path
    OUTPUT:
        None 
    '''
    #pickle.dump(model, open(model_filepath, 'wb'))
    joblib.dump(model,  model_filepath,compress=3)
    #joblib.dump(model, 'your_filename.pkl.z')   # zlib


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
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()