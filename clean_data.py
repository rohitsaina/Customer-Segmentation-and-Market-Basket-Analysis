import pandas as pd

# Load the Excel file
df = pd.read_excel('Online Retail.xlsx')

# Display initial info
print("Initial data shape:", df.shape)
print("Initial data types:")
print(df.dtypes)
print("Missing values:")
print(df.isnull().sum())

# Handle missing values
# Remove rows where UnitPrice is missing or zero (assuming invalid)
df = df[df['UnitPrice'].notna() & (df['UnitPrice'] > 0)]

# For CustomerID, fill missing with -1 (assuming guest customers)
df['CustomerID'] = df['CustomerID'].fillna(-1)

# Correct data types
# Convert InvoiceDate to datetime
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')

# Ensure Quantity and UnitPrice are numeric
df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
df['UnitPrice'] = pd.to_numeric(df['UnitPrice'], errors='coerce')

# Remove rows where Quantity or UnitPrice became NaN after conversion
df = df.dropna(subset=['Quantity', 'UnitPrice'])

# Remove duplicates
df = df.drop_duplicates()

# Display cleaned info
print("Cleaned data shape:", df.shape)
print("Cleaned data types:")
print(df.dtypes)
print("Missing values after cleaning:")
print(df.isnull().sum())

# Save the cleaned data
df.to_excel('Online_Retail_Cleaned.xlsx', index=False)
print("Cleaned data saved to 'Online_Retail_Cleaned.xlsx'")
