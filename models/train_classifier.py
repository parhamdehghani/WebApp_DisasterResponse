import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger','stopwords'])
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.metrics import confusion_matrix,f1_score,classification_report
import re
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, HashingVectorizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag, ne_chunk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.datasets import make_multilabel_classification
from sklearn.ensemble import RandomForestClassifier
import pickle


def load_data(database_filepath):
    '''
    Loads the saved database and generate inputs and labels for 
    model training step
    
    Input:
    database_filepath -> path to the saved database
    
    Output:
    X -> input data for model building
    Y -> labels to be used for model building
    cat_names -> column names of the labels used for model building
    
    '''
    
    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('Response',engine)
    X = df.iloc[:,1].values
    Y = df.iloc[:,4:].values
    cat_names = df.columns[4:]
    return X,Y,cat_names


def tokenize(text):
    '''
    Processes the input data by the text processing tools, then
    prepares input data for being transformed to input features
    for model building 
    
    Input:
    text -> a single document to be proceessed
    
    Output:
    clean_tokens -> processed document ready for feature transformation
    
    '''
    
    # removing punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    # split into words and remove stop words
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stopwords.words("english")]
    # lemmatize and normalize tokens
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''
    Builds the pipeline including text processing, feature transformation
    and grid search hyper-parameters for the used model
    
    Input:
    no input
    
    Output:
    model -> returns the whole pipeline ready for the training step
    
    '''
    # instantiating the multi-class predictor for the last step of pipeline
    mlpc = MultiOutputClassifier(estimator=RandomForestClassifier())
    # define the pipeline for ML workflow
    pipeline = Pipeline([('combining_feature',FeatureUnion([('initial_transformer',Pipeline([('bag_of_words',CountVectorizer(tokenizer=tokenize)),
                                                                                             ('tfidf_transformer',TfidfTransformer())]))])),
                         ('multi_classifier_building',mlpc)])
    # parameters to be evaluated in the defined pipeline
    parameters = {'multi_classifier_building__estimator__criterion': ['entropy'],
             'multi_classifier_building__estimator__n_estimators': [20],
             'combining_feature__initial_transformer__tfidf_transformer__sublinear_tf': [True]}
    # build the pipeline based on specified params
    model = GridSearchCV(pipeline, param_grid=parameters)
    # predicting the output for X_test data based on the best params
    return model



def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluates the trained model against the test data including accuracy
    of the model and precision and recall for each label
    
    Input:
    model -> generated model in the last step
    X_test -> test input data
    Y_test -> test date labels
    category_names -> column names of the labels
    
    Output:
    no return - prints the accuracy of the model against test data including
    precision and recall for each label of the test data
    
    '''
    # calculating the predictions to test set based on the trained model
    y_pred = model.predict(X_test)
    # calculating the accuracy for all the categories
    accuracy = (y_pred == Y_test).mean()
    print('Accuracy of the training is {} %\n'.format(accuracy*100))

    # reporting precision and recall for each column
    for col in range(y_pred.shape[1]):
        print('The result of evaluation of different categories are as follows:\n\n\n'+category_names[col]+':\n')
        print(classification_report(Y_test[:,col],y_pred[:,col],output_dict=False))
        print('\n')
    
def save_model(model, model_filepath):
    '''
    Saves a pickel file including the trained model with best set of 
    chosen hyper-parameters
    
    Input:
    model -> trained model
    model_filepath -> path to the pickle file to be saved
    
    Output:
    no return - saves the pickle file of the model referring to the 
    specified path to the model
    
    '''
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
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
