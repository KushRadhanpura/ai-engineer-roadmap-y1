# Project 2: Exploratory Data Analysis of Bookstore Data

This project focuses on cleaning, analyzing, and visualizing data from the `books.csv` dataset. The goal is to perform Exploratory Data Analysis (EDA) to uncover insights about the bookstore's inventory, such as price distribution and the relationship between book ratings and price.

---

## Analysis & Key Insights

The analysis was conducted in the `book_data_exploration.ipynb` Jupyter Notebook and involved several key steps:

* **Data Cleaning:**
    * Solved a critical text encoding error that incorrectly displayed the pound symbol '£' as 'Â£'.
    * Converted the `price` column from a text/object type to a numerical format (`float`) to enable calculations.
    * Ensured data integrity by confirming there were no missing price values after cleaning.

* **Key Discoveries:**
    1.  **Price Distribution:** The majority of books in the dataset are priced between **£20 and £55**.
    2.  **Price vs. Rating:** The analysis revealed that there is **no strong correlation** between a book's price and its star rating. High-rated books are not significantly more expensive than low-rated ones.
    3.  **Most Expensive Books:** The top 10 most expensive books in the catalogue were successfully identified and listed.

---

## How to Run This Analysis

To reproduce this analysis on your own machine, please follow these steps.

### 1. Prerequisites

You must have **Miniconda** or **Anaconda** installed on your system to manage the project's environment.

### 2. Set up the Conda Environment

Open your terminal, navigate to this project's directory (`02-data-analysis-report`), and run the following commands.

```bash
# Create a new conda environment named 'data-analysis' with Python 3.10
conda create --name data-analysis python=3.10

# Activate the newly created environment
conda activate data-analysis

```
 ### 3. Install Required Libraries

With the `data-analysis` environment active, install the necessary Python libraries.

```bash
# Install pandas for data manipulation, matplotlib/seaborn for plotting, and ipykernel for the notebook
pip install pandas matplotlib seaborn ipykernel


```
### 4. Run the Jupyter Notebook

Open the parent folder (ai-engineer-roadmap-y1) in Visual Studio Code.

Navigate to and open the book_data_exploration.ipynb file.

When prompted by VS Code, select the ('data-analysis': conda) kernel to connect the notebook to the environment you created.

You can now run the cells in the notebook to see the full analysis from start to finish.