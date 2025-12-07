import pandas as pd
import os
from sqlalchemy import create_engine
import logging
import time

# Configure logging
logging.basicConfig(
    filename='ingestion_db.log',
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='a'
)

# Database connection
engine= create_engine('sqlite:///customer_churn.db')

def ingest_db(df,table_name,engine):
    df.to_sql(table_name, con=engine, if_exists='replace', index=False)

# Function to load raw data and ingest into database
def load_raw_data():
    start_time = time.time()
    for file in os.listdir('dataset'):
        if '.csv' in file:
            logging.info(f'Starting ingestion for file: {file}')
            df = pd.read_csv('dataset/' + file)            
            logging.info(f'ingesion {file} in database')
            ingest_db(df,file[:-4],engine)
    end_time = time.time()
    total_time = (end_time - start_time)/60
    logging.info('Ingestion process completed.')
    logging.info(f'\nTotal time taken: {total_time} minutes')

# Main execution
if __name__ == '__main__':
    load_raw_data()
