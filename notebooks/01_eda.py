import pandas as pd

df = pd.read_csv('data/raw/Customer_support_data.csv')
print(df.columns)
print(df.info())
print(df.head())