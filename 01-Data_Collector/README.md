# Project 1: Automated Bookstore Analyst

This is an automated web scraping pipeline that gathers data from the "[Books to Scrape](http://books.toscrape.com/)" website.

## Features

- Navigates through all 50 pages of the catalogue automatically.
- Extracts the title, price, stock availability, star rating, and URL for all 1,000 books.
- Saves the clean, structured data into a `data/books.csv` file.

## How to Run

1.  **Set up the environment:**
    ```bash
    # Navigate to the project folder
    cd 01-Data_Collector

    # Create and activate a virtual environment
    python3 -m venv venv
    source venv/bin/activate

    # Install required packages
    pip install -r requirements.txt
    ```

2.  **Run the scraper:**
    ```bash
    python3 scraper.py
    ```