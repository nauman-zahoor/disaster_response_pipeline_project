import json
import sqlite3
import plotly
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Pie, Heatmap
from sqlalchemy import create_engine
import joblib
import re
# from sklearn.externals import joblib

app = Flask(__name__)


url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'


def tokenize(text):
    ''' Function that cleans text by first removing urls and then tokenizes,
        lemmatizes and lowers each token.
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


def load_data_from_db_and_create_db_if_not_exist(path):
    try:
        print('trying to find db')
        engine = create_engine('sqlite:///'+path)
        df = pd.read_sql_table('classification_history', engine)
        print('db found!')
    except:
        print('db not found... creating new!')
        conn = sqlite3.connect(path)
        # get a cursor
        cur = conn.cursor()
        # drop the test table in case it already exists
        cur.execute("DROP TABLE IF EXISTS classification_history")
        # create the test table including project_id as a primary key
        cur.execute("CREATE TABLE classification_history ( input_text TEXT, Output_Classification TEXT);")
        # insert a value into the classification_history table
        cur.execute('INSERT INTO classification_history (input_text, Output_Classification) VALUES ( "{}", "{}");'.format('', ''))
        #cur.execute('INSERT INTO classification_history (input_text, Output_Classification) VALUES ( "{}", "{}");'.format(query,  str(calassification_results)))
        conn.commit()
        # commit any changes and close the data base
        conn.close()
        df = pd.read_sql_table('classification_history', engine)
    return df


def save_classification_results(query,classification_labels,db_historical_classifications_path):
    try:
        conn = sqlite3.connect(db_historical_classifications_path)
        cur = conn.cursor()
        # drop the test table in case it already exists
        cur.execute('INSERT INTO classification_history (input_text, Output_Classification) VALUES ( "{}", "{}");'.format(query,  str(classification_labels)))
        conn.commit()      
        conn.close()
        return 1
    except:
        return 0



# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('data', engine)


# path of historical predictions db
db_historical_classifications_path = '../historic_predictions/historical_predictions.db'

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)



    # distribution of classes in data
    df_melted = df.melt(id_vars=['id', 'message','genre'],var_name='categories',
                        value_name='yes_no')
                           
    class_counts = df_melted[df_melted.yes_no==1].groupby('categories').count()['message'].sort_values(ascending=False)
    class_names = list(class_counts.index)
    
    # TODO: load data from historical_predictions.db. if not present, create one.
    # Data will be kept in table named classification_history
    # Table wil have attributes input_text:string and Output_Classification:string (string containing list of classifications)

    
    df_historical_classifications = load_data_from_db_and_create_db_if_not_exist(db_historical_classifications_path)
    
    #print('#############\n',df_historical_classifications,'\n')
    # TODO: extract count of classes from historical_predictions.db's data
    
    # df_historical_classifications
    df_historical_classifications_categories = pd.DataFrame([],columns = df.columns[4:])
    for i in range(len(df_historical_classifications)):
        vals = df_historical_classifications.loc[i,'Output_Classification']
        if len(vals)>0:
            #print('$$$$$$$$$  ',vals)
            vals = [int(val.strip()) for val in vals.replace('[','').replace(']','').split(' ')]
            df_historical_classifications_categories.loc[i,:] = vals
    temp = df_historical_classifications_categories.sum().sort_values(ascending=False)
    historical_classifications_counts = temp
    historical_classifications_names = list(temp.index)
    
    
    #extract data counts that is originally in english vs translated
    englist_nonenglish = df.loc[:,['message','original']].notna().sum()
    englist_nonenglish.index = ['English','Other']
    englist_nonenglish[0] = englist_nonenglish[0]-englist_nonenglish[1]
    englist_nonenglish_counts = englist_nonenglish
    englist_nonenglish_names = list(englist_nonenglish.index)
    
    
    # corelation of output categories
    df_cats = df[df.columns[4:]]
    corr = df_cats.corr()
    
    corr_values = corr.values
    corr_names = list(corr.index)

    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Pie(
                    labels=genre_names,
                    values=genre_counts
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
                    x=class_names,
                    y=class_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Output Categoreis',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Categories"
                }
            }
        },
        
         {
            'data': [
                Pie(
                    labels=englist_nonenglish_names,
                    values=englist_nonenglish_counts
                )
            ],

            'layout': {
                'title': 'English vs Non English messages in training data',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Language"
                }
            }
        },

        # corr plot
         {
            'data': [
                Heatmap(
                    x=corr_names,
                    y = corr_names,
                    z=corr_values,
                )
            ],

            'layout': {
                'title': 'Features Correlation Matrix',
                'width': 1200,
                 'height': 900,
              
            }
        },
        # TODO: Add graph for historical prediction counts
         {
            'data': [
                Bar(
                    x=historical_classifications_names,
                    y=historical_classifications_counts
                )
            ],

            'layout': {
                'title': 'Counts of Historical Classification Categories',
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
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    #print(ids)
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
    #print(classification_labels)

    # TODO: save query and classification_results in classification table of historical_predictions.db
    if save_classification_results(query, classification_labels, db_historical_classifications_path):
        print('classification results saved to db')
    else:
        print('error!...couldnt save classification results saved to db')

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




