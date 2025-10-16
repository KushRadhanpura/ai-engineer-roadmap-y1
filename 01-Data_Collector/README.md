# Project 1: Automated Bookstore Scraper

This is an automated web scraping pipeline that gathers data from the "[Books to Scrape](http://books.toscrape.com/)" website using Python's `requests` and `BeautifulSoup` libraries.

---

## Prerequisites

- Python 3.x

---

## Features

- Navigates through all 50 pages of the catalogue automatically.
- Extracts the title, price, stock availability, star rating, and URL for all 1,000 books.
- Saves the clean, structured data into a `books.csv` file.

---

## How to Run

### 1. Set up the Environment

First, navigate to the project folder and set up a Python virtual environment.

```bash
# Navigate to the project folder
cd 01-Data_Collector

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install all required packages from the requirements file
pip install -r requirements.txt


```
### 2.Run the Scraper

```bash
# Execute the scraper script
python3 scraper.py