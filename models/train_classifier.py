import sys
import pandas as pd
import re
import pickle
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier


def load_data(database_filepath):
    '''
    Input:
    database_filepath
    
    Output:
    X - independent series values
    Y - dependent dataframe values
    '''
    # create a processed sqlite db
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('table',engine)  
    X = df['message']
    Y = df.drop(columns=['id','genre','message','original'],axis=1)
    return X,Y

def tokenize(text):
    '''
    Input:
    string text
    
    Output:
    list of string
    
    ex: 
    Input -> 'Weather update - a cold front from Cuba that'
    Output -> ['weather', 'update', 'cold', 'front', 'cuba']
    '''
    # substitute the symbols wich are not a-zA-Z0-9
    text = re.sub(r"[^a-zA-Z0-9]"," ",text)
    # tokenize
    words = word_tokenize(text)
    # remove the stopwords
    words = [w for w in words if w not in stopwords.words("english")]
    # lemmatize put the tokens  list
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in words:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    
    return clean_tokens


def build_model():
    '''
    Input: 
    None
    
    Output:
    sklearn model
    '''
    
    pipeline = Pipeline([
    ('vect',CountVectorizer(tokenizer=tokenize)),
    ('tfidf',TfidfTransformer()),
    ('clf',MultiOutputClassifier(AdaBoostClassifier())),
    ])
    # paramters used for gridsearch
    parameters = {'tfidf__use_idf': (True, False), 
              'clf__estimator__n_estimators': [20,80,100],
              'clf__estimator__base_estimator': [SVC(), DecisionTreeClassifier()]
              }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, Y_test):
    ''' 
    Input:
    sklearn model, X, Y test data
    
    Output:
    None
    '''
    for i, col in enumerate(Y_test):
        predicted = model.predict(X_test)
        print(classification_report(Y_test[col],predicted[:, i]))

def save_model(model, model_filepath):
    '''
    Input
    model, model_filepath
    
    Output:
    pickle file
    '''
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

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