import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np

# Load the Excel file
df = pd.read_excel('Online Retail.xlsx')

# Data Cleaning (same as before)
df = df[df['UnitPrice'].notna() & (df['UnitPrice'] > 0)]
df['CustomerID'] = df['CustomerID'].fillna(-1)
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
df['UnitPrice'] = pd.to_numeric(df['UnitPrice'], errors='coerce')
df = df.dropna(subset=['Quantity', 'UnitPrice'])
df = df.drop_duplicates()

# Descriptive statistics for numerical columns
desc_stats = df.describe()
print("Descriptive Statistics:")
print(desc_stats)

# Frequency analysis: Top 10 countries by number of transactions
country_counts = df['Country'].value_counts().head(10)
print("\nTop 10 Countries by Number of Transactions:")
print(country_counts)

# Plot top 10 countries
plt.figure(figsize=(10,6))
sns.barplot(x=country_counts.values, y=country_counts.index, palette='viridis')
plt.title('Top 10 Countries by Number of Transactions')
plt.xlabel('Number of Transactions')
plt.ylabel('Country')
plt.tight_layout()
plt.savefig('top_countries.png')
plt.close()

# Frequency analysis: Top 10 products by quantity sold
product_counts = df.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10)
print("\nTop 10 Products by Quantity Sold:")
print(product_counts)

# Plot top 10 products
plt.figure(figsize=(10,6))
sns.barplot(x=product_counts.values, y=product_counts.index, palette='magma')
plt.title('Top 10 Products by Quantity Sold')
plt.xlabel('Quantity Sold')
plt.ylabel('Product Description')
plt.tight_layout()
plt.savefig('top_products.png')
plt.close()

# Time-series analysis: Daily sales revenue
df['Sales'] = df['Quantity'] * df['UnitPrice']
daily_sales = df.groupby(df['InvoiceDate'].dt.date)['Sales'].sum()

print("\nDaily Sales Revenue (first 10 days):")
print(daily_sales.head(10))

# Plot daily sales revenue
plt.figure(figsize=(12,6))
daily_sales.plot()
plt.title('Daily Sales Revenue')
plt.xlabel('Date')
plt.ylabel('Sales Revenue')
plt.tight_layout()
plt.savefig('daily_sales.png')
plt.close()

# RFM Analysis for Customer Segmentation
# Recency: days since last purchase
snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
    'InvoiceNo': 'nunique',
    'Sales': 'sum'
})
rfm.rename(columns={'InvoiceDate': 'Recency', 'InvoiceNo': 'Frequency', 'Sales': 'Monetary'}, inplace=True)

print("\nRFM Table (first 10 customers):")
print(rfm.head(10))

# Save RFM table to CSV
rfm.to_csv('rfm_table.csv')

# Market Basket Analysis
from mlxtend.frequent_patterns import apriori, association_rules

# Create basket matrix
basket = df.groupby(['InvoiceNo', 'Description'])['Quantity'].sum().unstack().reset_index().fillna(0).set_index('InvoiceNo')
basket = basket.applymap(lambda x: 1 if x > 0 else 0)

# Frequent itemsets
frequent_itemsets = apriori(basket, min_support=0.01, use_colnames=True)
print("\nFrequent Itemsets (min support 0.01):")
print(frequent_itemsets.head(10))

# Association rules
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules = rules.sort_values('lift', ascending=False)
print("\nTop Association Rules by Lift:")
print(rules.head(10))

# Save rules to CSV
rules.to_csv('association_rules.csv')

print("\nEDA analysis completed. Plots saved as PNG files, RFM table and association rules saved as CSV.")
