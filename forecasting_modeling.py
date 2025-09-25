import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from statsmodels.tsa.arima.model import ARIMA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load and clean data
df = pd.read_excel('Online Retail.xlsx')
df = df[df['UnitPrice'].notna() & (df['UnitPrice'] > 0)]
df['CustomerID'] = df['CustomerID'].fillna(-1)
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
df['UnitPrice'] = pd.to_numeric(df['UnitPrice'], errors='coerce')
df = df.dropna(subset=['Quantity', 'UnitPrice'])
df = df.drop_duplicates()
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

# Sales Forecasting
# Aggregate monthly sales
df['Month'] = df['InvoiceDate'].dt.to_period('M')
monthly_sales = df.groupby('Month')['TotalPrice'].sum()
monthly_sales.index = monthly_sales.index.to_timestamp()

# Fit ARIMA model
model = ARIMA(monthly_sales, order=(5,1,0))
model_fit = model.fit()
forecast = model_fit.forecast(steps=12)
print("Sales Forecasting (next 12 months):")
print(forecast)

# Plot forecast
plt.figure(figsize=(10,6))
plt.plot(monthly_sales, label='Historical Sales')
plt.plot(forecast, label='Forecast', color='red')
plt.title('Sales Forecasting')
plt.legend()
plt.savefig('sales_forecast.png')
plt.close()

# Clustering: Group customers using KMeans on RFM features
snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
    'InvoiceNo': 'nunique',
    'TotalPrice': 'sum'
}).rename(columns={'InvoiceDate': 'Recency', 'InvoiceNo': 'Frequency', 'TotalPrice': 'Monetary'})

# Normalize RFM for clustering
rfm_normalized = (rfm - rfm.mean()) / rfm.std()

# KMeans clustering
kmeans = KMeans(n_clusters=4, random_state=42)
rfm['Cluster'] = kmeans.fit_predict(rfm_normalized)
print("\nCustomer Clusters:")
print(rfm['Cluster'].value_counts())

# Save clustered customers
rfm.to_csv('customer_clusters.csv')

# Churn Prediction
# Define churn: no purchase in last 90 days
last_date = df['InvoiceDate'].max()
churn_threshold = last_date - pd.Timedelta(days=90)
churn_df = df.groupby('CustomerID').agg({'InvoiceDate': 'max'})
churn_df['Churn'] = (churn_df['InvoiceDate'] < churn_threshold).astype(int)

# Merge with RFM
churn_data = rfm.merge(churn_df[['Churn']], left_index=True, right_index=True)

# Features and target
X = churn_data[['Recency', 'Frequency', 'Monetary']]
y = churn_data['Churn']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Logistic Regression
clf = LogisticRegression()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("\nChurn Prediction Report:")
print(classification_report(y_test, y_pred))

print("\nForecasting and Modeling completed. Files saved: sales_forecast.png, customer_clusters.csv.")
