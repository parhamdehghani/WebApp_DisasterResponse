# Disaster Response Pipeline 
# Written by Parham Dehghani
# Contact: parham.dehghani@concordia.ca


This is the codes relevant to the NLP classifier based on tfidf transformer and including a whole pipeline
starting from text processing then transforming into features, training the model based on Random Forest 
classification algorithm and then deployment on the fron-end using Flask. Fron-end is a web application in
which the message in the event of a disaster is elicited, and then classification will be done based on the 
trained model at the backend of the webapp. These are the simple instructions how to set up and run the web app.
This web app can be simply deployed on REST endpoints of AWS or any other cloud-based services as the extension. 

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run the web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
