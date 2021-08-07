# importing libraries
import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    input:
        -messages file path
        -categories file path
    output:
        -combined data frame
    """

    # reading files
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # merges dataframes based on their IDs
    df = pd.merge(messages, categories, on=['id'])

    return df


def clean_data(df):
    """
    input:
        - uncleaned combined dataframe
    output:
        - cleaned combined dataframe
    """

    # creating column for different categories
    categories = df['categories'].str.split(';', expand=True)
    # retrieve column names
    category_col_names = [col.split('-')[0] for col in categories.loc[0]]
    # respecify the column names and reg the value
    categories.columns = category_col_names
    # clean data, that is, leave last number alone
    for col in categories.columns:
        categories[col] = categories[col].apply(lambda x: x[-1])
        categories[col] = categories[col].astype(int)
    # for sure there is no not 1 or 0 data
    for col in categories.columns:
        bool_1 = (categories[col] != 0)
        bool_2 = (categories[col] != 1)
        categories.loc[bool_1 & bool_2, col] = 1
    # drop redundant col
    df.drop(labels=['categories'], axis=1, inplace=True)
    # re-merge data
    df = pd.concat([df, categories], axis=1)
    # drop the duplicates
    df.drop_duplicates(inplace=True)

    return df


def save_data(df, database_filename):
    """
    input:
        - clean combined data
        - the name of the database
    this function will save dataframe to a sql database
    """
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('dis_res', engine, index=False, if_exists='replace')


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
        print('Please provide the filepaths of the messages and categories ',
              'datasets as the first and second argument respectively, as ',
              'well as the filepath of the database to save the cleaned data ',
              'to as the third argument. \n\nExample: python process_data.py ',
              'disaster_messages.csv disaster_categories.csv ',
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
