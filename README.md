# Disaster Response Pipeline Project

## Installation
Install required libraries using requirements file using the following instruction.
```
pip install -r requirements.txt 
```
## Instructions
Following commands when ran in the project's root directory will set up database and model.

Run following to run ETL pipeline that cleans data and stores in database
```
python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
```

Run following to run Machine learning pipeline that trains classifier and saves model
```
python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
```
To run the webapp, run the following command in the app's directory.
```
python run.py
```
By Default the app runs at port 3001 and can be accessed at http://0.0.0.0:3001/

## Fearures:
### ETL:
The ETL pipeline takes two datasets from [Figure Eight](https://www.figure-eight.com/). It cleans and the join the data. Pipeline then loads data to a database.

### Machine learning:
Machine learning pipeline extracts data from db and then splits data in training and test sets.
Model is then initialised. Model consists of a pipeline that first transforms data using CountVectorizer and then TfidfTransformer. Classifier is RandomForestClassifier encapsulated in MultiOutputClassifier for multi-class classification. A list of model parameters is defined and GridSearchCV is applied to search for best parameters for model.
Model is then trained on trianing data and evaluated on test data and model is saved.

### Webapp:


### Heroku deployment:
