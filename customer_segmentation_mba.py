import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules

# Load and clean data
df = pd.read_excel('Online Retail.xlsx')
df = df[df['UnitPrice'].notna() & (df['UnitPrice'] > 0)]
df['CustomerID'] = df['CustomerID'].fillna(-1)
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
df['UnitPrice'] = pd.to_numeric(df['UnitPrice'], errors='coerce')
df = df.dropna(subset=['Quantity', 'UnitPrice'])
df = df.drop_duplicates()

# Create TotalPrice
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

# Customer Segmentation: RFM Analysis
snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,  # Recency
    'InvoiceNo': 'nunique',  # Frequency
    'TotalPrice': 'sum'  # Monetary
}).rename(columns={'InvoiceDate': 'Recency', 'InvoiceNo': 'Frequency', 'TotalPrice': 'Monetary'})

# Assign RFM scores (1-5 scale, 5 being best)
rfm['R_Score'] = pd.qcut(rfm['Recency'], 5, labels=[5, 4, 3, 2, 1])
rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
rfm['M_Score'] = pd.qcut(rfm['Monetary'], 5, labels=[1, 2, 3, 4, 5])

# Combine scores
rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)

# Segment customers
def segment_customer(row):
    if row['RFM_Score'] in ['555', '554', '545', '544', '455', '454', '445']:
        return 'High-Value Customers'
    elif row['RFM_Score'] in ['511', '521', '531', '541', '551']:
        return 'New Customers'
    elif row['RFM_Score'] in ['115', '125', '135', '145', '155']:
        return 'At-Risk Customers'
    elif row['RFM_Score'] in ['111', '112', '113', '114', '121', '122', '123', '124']:
        return 'Lost Customers'
    else:
        return 'Regular Customers'

rfm['Segment'] = rfm.apply(segment_customer, axis=1)

print("RFM Analysis Summary:")
print(rfm.head(10))
print("\nSegment Counts:")
print(rfm['Segment'].value_counts())

# Save RFM
rfm.to_csv('rfm_segmentation.csv')

# Plot segments
segment_counts = rfm['Segment'].value_counts()
plt.figure(figsize=(10,6))
plt.bar(segment_counts.index, segment_counts.values)
plt.title('Customer Segments')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('customer_segments.png')
plt.close()

# Market Basket Analysis
basket = df.groupby(['InvoiceNo', 'Description'])['Quantity'].sum().unstack().fillna(0)
basket = basket.applymap(lambda x: 1 if x > 0 else 0)

frequent_itemsets = apriori(basket, min_support=0.01, use_colnames=True)
print("\nFrequent Itemsets:")
print(frequent_itemsets.head(10))

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
rules = rules.sort_values('lift', ascending=False)
print("\nTop Association Rules:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))

# Save rules
rules.to_csv('market_basket_rules.csv')

print("\nAnalysis completed. Files saved: rfm_segmentation.csv, customer_segments.png, market_basket_rules.csv")
