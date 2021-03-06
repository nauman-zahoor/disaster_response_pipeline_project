# Disaster Response Pipeline Project
App is live at: https://disaster-relief-nauman.herokuapp.com

## Motivation
Motivation behind this project is to learn how a machine learning project for real life scenario is developed from start i.e. collecting data to delpoying it. 
This project builds a machine learning classifier and deploys it on a webapp. Webapp uses model and classifies incoming messages (in our secnario: disaster realted) in to categories like Water,Refugees,Floods etc. so that disaster relief workers can identify who needs what automatically.


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
Webapp's main page displays charts from Training set and Historical classifications as discussed below.
#### From Training set
From the Training set, app displays:
1. Distribution of message genres present in training data.
![](https://github.com/nauman-zahoor/disaster_response_pipeline_project/blob/main/images/distribution_of_message_genres.png?raw=true)
2. English vs Non English messages in training data.
![](https://github.com/nauman-zahoor/disaster_response_pipeline_project/blob/main/images/English_vs_nonenglish_messages.png?raw=true)
3. Distribution of Output Categoreis.
![](https://github.com/nauman-zahoor/disaster_response_pipeline_project/blob/main/images/output_category_distribution.png?raw=true)
4. Features Correlation Matrix
![](https://github.com/nauman-zahoor/disaster_response_pipeline_project/blob/main/images/fearures_correlation.png?raw=true)

 
#### From Historical Predictions
Another functionality added to webapp is the ability to store previous classified messages and the corrosponding classification results. Both these are stored in a seperate db named historical_predictions.db in historic_predictions directory. Whenever someone classifies a message, input message and resutls are stored. 
Whenever the index page is loaded, app reads from historic_predictions db and extracts the counts of previous classification categories and plots these. 
![](https://github.com/nauman-zahoor/disaster_response_pipeline_project/blob/main/images/historical_prediction_categories.png?raw=true)


## App Screenshots
App main page:

![Mani page](https://github.com/nauman-zahoor/disaster_response_pipeline_project/blob/main/images/webapp_index-page.png?raw=true)

App Classification page:

![Classification page](https://github.com/nauman-zahoor/disaster_response_pipeline_project/blob/main/images/webapp_classification_page.png?raw=true)

## Authors

* [Nauman Zahoor](https://github.com/nauman-zahoor/)

## License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Acknowledgements

* [Udacity](https://www.udacity.com/)
* [Figure Eight](https://www.figure-eight.com/) 

