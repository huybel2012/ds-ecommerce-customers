import pandas as pd
df = pd.read_csv('data/raw/ecommerce_customer_data_custom_ratios.csv')
print(df.columns)
print(df.info())
print(df.head())