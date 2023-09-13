import pandas as pd
from sqlalchemy import create_engine
import sys



def data_loading(input_file1, input_file2):
    """
    Load the data
    
    Input:
    input_file1 - path to CSV file containing messages
    input_file2 - path to CSV file containing categories
    
    Output:
    merged_df - Merged data from input files
    """    
    messages = pd.read_csv(input_file1)
    categories = pd.read_csv(input_file2)
    merged_df = pd.merge(messages, categories, on='id')
    
    return merged_df
    

def data_wraNgling(data_df):
    """
    Cleansing the data
    
    Input:
    data_df - Merged data containing messages and categories
    
    Output:
    cleaned_df - Processed and cleaned data
    """
    # split the data and get new column as header
    category = data_df['categories'].str.split(pat=';', expand=True)
    new_columns = [str(x).split('-')[0] for x in category.iloc[0]]
    # rename the columns of `categories`
    category.columns = new_columns
    # replace value to 0 or 1
    categories_transformed = category.map(lambda x: int(x.split('-')[-1]))
    # drop the original categories column from `df`
    data_df = data_df.drop('categories', axis = 1)
    # concat to df
    data_df = pd.concat([data_df, categories_transformed], axis =1)
    # drop duplicate
    data_df.drop_duplicates(inplace = True)
    # change value of df.related that contain value 2
    data_df.loc[data_df['related']==2, 'related'] = 1
    # clean up drop column contain only null
    cleaned_df = data_df.drop('child_alone', axis=1)

    return cleaned_df

def save_data_to_table(cleaned_df, db_filename):
    """
    Input:
    cleaned_df - Cleaned data containing messages and categories
    db_filename - Filename for SQLite database
    
    Output:
    None - Saves the cleaned data to an SQLite database
    """
    engine = create_engine('sqlite:///' + db_filename)
    cleaned_df.to_sql('DisasterResponse_table', engine, index=False, if_exists='replace')

def main():
    """
    Orchestrates the data processing pipeline.
    
    Input:
    None
    
    Output:
    None - Saves cleaned data to a database
    
    This function orchestrates the entire data processing pipeline. It loads messages
    and categories data from CSV files, performs cleaning and processing, and saves
    the cleaned data to an SQLite database for further analysis.
    """
    if len(sys.argv) == 4:
        input_file1, input_file2, db_filename = sys.argv[1:]

        print('Loading data...\n    INPUT FILE 1: {}\n    INPUT FILE 2: {}'.format(input_file1, input_file2))
        data_df = data_loading(input_file1, input_file2)
        print('Cleaning data...')
        cleaned_df = data_wraNgling(data_df)
        print('Saving data...\n    DATABASE: {}'.format(db_filename))
        save_data_to_table(cleaned_df, db_filename)
        print('Data processing completed and saved!')
    
    else:
        print('Please provide the paths of the input data files and the output database file as command-line arguments.\n'
              'For example: python process_data.py input_messages.csv input_categories.csv output_database.db')

if __name__ == '__main__':
    main()
