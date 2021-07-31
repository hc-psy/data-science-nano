# importing libraries
import sys
import pandas as pd
from sqlalchemy import create_engine


def read_csv_data(messages_filepath, categories_filepath):
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


def clean_data_frame(df):
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


def sql_save(df, database_filename):
    """
    input:
        - clean combined data
        - the name of the database
    this function will save dataframe to a sql database
    """
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('des_res', engine, index=False, if_exists='replace')


def main():

    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filename = sys.argv[1:]

        print('Loading data...')
        df = read_csv_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data_frame(df)

        print('Saving sql data...')
        sql_save(df, database_filename)

        print('Finished and have a check!')

    else:
        print('Please properly refer the input instructions in the repo')


if __name__ == '__main__':
    main()
