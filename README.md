# Disaster Response Pipeline 
# Written by Parham Dehghani
# Contact: parham.dehghani@concordia.ca


This project involves the NLP classifier based on tfidf transformer and including a whole pipeline
starting from text processing then transforming into features (the whole ETL pipeline), training the model 
based on Random Forest classification algorithm and searching over chosen hyperparameters of the model and 
then deployment on the fron-end using Flask. Fron-end is a web application in which the message in the event 
of a disaster is elicited, and then classification will be done based on the trained model at the backend of 
the webapp. 

In this incarnaton, this project can be beneficial to be expanded for the commercial use in the event 
of a disaster as it can help inform the regarding sectors as fast as possible and then add to the performance to
the ongoing endeavors for resolving issues. By classifying the upcoming messages over the social media, the relevant
officials can be better and faster informed about the problems and the level by which people are being impacted. 


These are the simple instructions how to set up and run the web app. This web app can be simply deployed on REST endpoints 
of AWS or any other cloud-based services as the extension. The input data can also be provided by the relevant APIs to add
to the precision of the classifier as the trend within data can change over time that can impact the working ranges for the 
hyperparameters o the chosen algorithm.
 

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run the web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


### File structure:
app:

- template # including html files of the designed webpages

	-- master.html # main page of web app including insights about the used data

	-- go.html # classification result page of web app

- run.py # Flask file that runs app

data:

- disaster_categories.csv # data to process
- disaster_messages.csv # data to process
- process_data.py # code to run the whole ETL pipeline upon the input data
- InsertDatabaseName.db # database to save clean data to

models:

- train_classifier.py # code to run the training job based on the chosen specifications
- classifier.pkl # saved model

README.md
