# import libraries
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Loads data from input files and merges them to result in
    a final dataframe for further process

    Input:
    
    messages_filepath -> path to messages.csv
    categories_filepath -> path to categories.csv

    Output:
    
    df -> final dataframe including both messages and categories data

    '''
    # load datasets
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = pd.merge(messages,categories,on='id')
    # create new columns based on response categories
    categories = df['categories'].str.split(';',expand=True)
    row = categories.iloc[0,:]
    category_colnames = row.apply(lambda x: x.split('-')[0])
    categories.columns = category_colnames
    for column in categories.columns:
        categories[column] = categories[column].apply(lambda x: int(x.split('-')[1]))
    df.drop(['categories'],axis=1,inplace=True)
    # finalize dataframe based on new columns for response categories
    df = pd.concat([df,categories],axis=1)
    return df


def clean_data(df):
    '''
    Cleans data from any duplicates in two columns message and original
    
    Input:
    df -> generated dataframe in the previous step
    
    Output:
    df -> cleaned dataframe
    
    '''
    
    # drop duplicates based on 'message' and 'original' columns
    df.drop_duplicates(subset=['message','original'],inplace=True)
    return df


def save_data(df, database_filename):
    '''
    Saves dataframe to a database
    
    Input:
    df -> generated dataframe in the previous step
    database_filename -> arbitrary name for the saved database
    
    Output:
    no output - function saves a database including Response table
    
    '''

    # create a database including prcessed dataframe
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('Response', engine, index=False)
    pass


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
