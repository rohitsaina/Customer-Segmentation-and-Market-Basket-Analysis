import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Load data
df = pd.read_excel('Online Retail.xlsx')

# Data preprocessing for Market Basket Analysis
# Create one-hot encoded matrix with InvoiceNo as index and Description as columns
basket = pd.crosstab(df['InvoiceNo'], df['Description'])

# Convert values to 1s and 0s to indicate presence or absence of an item
basket = basket.where(basket == 0, 1)

print("Basket matrix shape:", basket.shape)

# Apply Apriori algorithm to find frequent itemsets with min_support=0.01
frequent_itemsets = apriori(basket, min_support=0.01, use_colnames=True)
print("\nFrequent Itemsets (min support=0.01):")
print(frequent_itemsets.head(10))

# Generate association rules with metric='lift' and min_threshold=1
rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1)
rules = rules.sort_values(by='lift', ascending=False)

print("\nTop 10 Association Rules by Lift:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))

# Save frequent itemsets and rules to CSV files
frequent_itemsets.to_csv('frequent_itemsets.csv', index=False)
rules.to_csv('association_rules.csv', index=False)

print("\nMarket Basket Analysis completed. Results saved to 'frequent_itemsets.csv' and 'association_rules.csv'.")
