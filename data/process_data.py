import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Input:
    messages_filepath, categories_filepath
    
    Output:
    dataframe merged by two raw data
    '''
    messages = pd.read_csv(messages_filepath)
    categories =  pd.read_csv(categories_filepath)
    df = pd.merge(messages,categories,on='id')
    
    return df

def clean_data(df):
    ''' clean the raw dataframe
    Input:
    dataframe
    
    Output:
    dataframe
    '''
    # take the 36 columns from dataframe, and make a new dataframe categories
    categories = pd.DataFrame(df.categories.str.split(';',expand=True))
    # take a first row, than rename the columns of dataframe categories
    row = categories.loc[0]
    category_colnames = row.apply(lambda x:x[:-2])
    categories.columns = category_colnames
    # loop the dataframe categories, and Convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda col: col.split('-')[1])
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    # there are  0,1,2 in columns=related, need to exlude the 2
    categories = categories[categories.related.isin([2]) == False]
    # drop the original the columnn of categories from raw dataframe, then merge with dataframe categories
    df.drop(columns='categories',axis=1,inplace=True)
    df = pd.concat([df,categories],axis=1)
    df.drop_duplicates(subset=['message','original'],inplace=True)
    
    return df

def save_data(df, database_filename):
    '''
    Input:
    dataframe
    
    Output:
    database_filename
    '''
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('table', engine, index=False, if_exists='replace')


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