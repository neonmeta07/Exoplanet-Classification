import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


sns.set(style='whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

#Loading dataset
df = pd.read_csv('exoplanet.csv')

#Initial Exploration
print("First few rows:")
print(df.head())

print("\n Dataset Shape:", df.shape)
print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\n📉 Summary Statistics:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum().sort_values(ascending=False).head(15))
