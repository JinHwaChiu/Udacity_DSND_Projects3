# Disaster Response Pipeline Project

### Instructions:

This is a project to analyze disaster data from Appen (formally Figure 8) than classifies disaster messages.  
From web app, you can type in any messages at searching block, then there will be classification table shown at below

![image info](pic1.png)



### How to use ?:  
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
