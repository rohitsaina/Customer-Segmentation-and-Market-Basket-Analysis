import pandas as pd
import os

def test_cleaning():
    # Load original and cleaned data
    original_df = pd.read_excel('Online Retail.xlsx')
    cleaned_df = pd.read_excel('Online_Retail_Cleaned.xlsx')

    print("Original data shape:", original_df.shape)
    print("Cleaned data shape:", cleaned_df.shape)

    # Check missing values in cleaned
    missing = cleaned_df.isnull().sum().sum()
    print("Total missing values in cleaned data:", missing)
    assert missing == 0, "Cleaned data should have no missing values"

    # Check data types
    assert pd.api.types.is_datetime64_any_dtype(cleaned_df['InvoiceDate']), "InvoiceDate should be datetime"
    assert pd.api.types.is_numeric_dtype(cleaned_df['Quantity']), "Quantity should be numeric"
    assert pd.api.types.is_numeric_dtype(cleaned_df['UnitPrice']), "UnitPrice should be numeric"

    # Check no zero or negative UnitPrice
    assert (cleaned_df['UnitPrice'] > 0).all(), "UnitPrice should be positive"

    print("Cleaning tests passed.")

def test_eda():
    # Check if files exist
    files = ['rfm_table.csv', 'top_countries.png', 'top_products.png', 'daily_sales.png']
    for file in files:
        assert os.path.exists(file), f"{file} should be created"
    print("EDA output files exist.")

    # Load RFM table
    rfm = pd.read_csv('rfm_table.csv', index_col=0)
    print("RFM table shape:", rfm.shape)
    assert 'Recency' in rfm.columns, "RFM should have Recency"
    assert 'Frequency' in rfm.columns, "RFM should have Frequency"
    assert 'Monetary' in rfm.columns, "RFM should have Monetary"

    print("EDA tests passed.")

if __name__ == "__main__":
    test_cleaning()
    test_eda()
    print("All tests passed!")
