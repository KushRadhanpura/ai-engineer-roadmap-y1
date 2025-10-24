import psycopg2
import pandas as pd
from io import StringIO
import os
import csv 
from urllib.parse import urlparse 

RENDER_DB_HOST = os.environ.get("RENDER_DB_HOST")
RENDER_DB_NAME = os.environ.get("RENDER_DB_NAME")
RENDER_DB_USER = os.environ.get("RENDER_DB_USER")
RENDER_DB_PASS = os.environ.get("RENDER_DB_PASS")
DATABASE_URL = os.environ.get("DATABASE_URL") 


LOCAL_CSV_PATH = "/home/kushsoni/Desktop/ai-engineer-roadmap-y1/02-data-analysis-report/books_clean.csv" 


TABLE_NAME = "books_data" 


def upload_csv_to_postgres(file_path, table_name, conn_string):
    """Connects to the cloud DB and uploads the local CSV."""
    conn = None
    cur = None
    
    print(f"--- STARTING DATA MIGRATION ---")
    
    try:
        
        if conn_string.startswith("postgresql://"):
            conn = psycopg2.connect(conn_string)
        else:
            
            conn = psycopg2.connect(conn_string)
            
        conn.autocommit = True
        cur = conn.cursor()
        print(f"2. Connection successful. Preparing data upload...")

        df = pd.read_csv(file_path)
        print(f"3. Local CSV loaded: {len(df)} rows.")

        cur.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE;")
        
        
        create_table_sql = pd.io.sql.get_schema(df, table_name)
        cur.execute(create_table_sql)
        print(f"4. Table {table_name} recreated on cloud DB.")

       
        buffer = StringIO()
        
        
        df.to_csv(buffer, index=False, header=False, sep='\t', quoting=csv.QUOTE_MINIMAL)
        
        buffer.seek(0)
        
        
        cur.copy_from(buffer, table_name, sep="\t", null='') 
        
        cur.execute(f"SELECT COUNT(*) FROM {table_name};")
        count = cur.fetchone()[0]
        
        print(f"\n***** SUCCESS! {count} ROWS INSERTED. *****")
        print("Quarter 1 Deployment Data is ready.")

    except FileNotFoundError:
        print("\nFATAL ERROR: Local data file NOT FOUND.")
        print(f"Check the path: {file_path}")
    except Exception as e:
        print("\n***** FATAL CONNECTION/UPLOAD ERROR *****")
        print(f"ERROR DETAILS: {e}")
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()
        print("--- MIGRATION SCRIPT FINISHED ---")


if __name__ == "__main__":
    
    if DATABASE_URL:
        
        CONN_STRING = DATABASE_URL
    elif RENDER_DB_HOST and RENDER_DB_PASS:
        
        CONN_STRING = (
            f"dbname={RENDER_DB_NAME} user={RENDER_DB_USER} password={RENDER_DB_PASS} "
            f"host={RENDER_DB_HOST} sslmode=require" 
        )
    else:
        print("FATAL ERROR: No DATABASE_URL or RENDER_DB_* environment variables found.")
        CONN_STRING = None

    if CONN_STRING:
        upload_csv_to_postgres(LOCAL_CSV_PATH, TABLE_NAME, CONN_STRING)