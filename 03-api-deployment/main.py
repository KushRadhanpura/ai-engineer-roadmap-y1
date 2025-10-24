# main.py
from fastapi import FastAPI, HTTPException
import psycopg2
import os


DB_HOST = os.getenv("DB_HOST", "localhost")
DB_NAME = os.getenv("DB_NAME", "bookstore_db")
DB_USER = os.getenv("DB_USER", "postgres")    
DB_PASS = os.getenv("DB_PASS", "kush1675") 


app = FastAPI(title="Quarter 1 Data API")


def get_db_connection():
    """Establishes a connection to the PostgreSQL database."""
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASS
        )
        return conn
    except Exception as e:
        print(f"ERROR: Database connection failed: {e}")
        
        return None



@app.get("/", tags=["Health Check"])
def read_root():
    """Simple health check endpoint."""
    return {"message": "Data API is running. Visit /docs for documentation."}

@app.get("/data/record_count", tags=["Data"])
def get_data_count():
    """Fetches the total number of records from your primary data table."""
    conn = get_db_connection()
    if conn is None:
        
        raise HTTPException(status_code=503, detail="Database connection unavailable.")

    try:
        cur = conn.cursor()
        
        cur.execute("SELECT COUNT(*) FROM books_data;")
        count = cur.fetchone()[0]
        cur.close()
        return {"data_source": DB_NAME, "record_count": count}

    except Exception as e:
        print(f"ERROR: Query failed: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving data from the database.")
    
    finally:
        if conn:
            conn.close()