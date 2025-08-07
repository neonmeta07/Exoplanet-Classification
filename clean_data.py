import pandas as pd

df = pd.read_csv('exoplanet.csv')

# Dropping columns with more than 40% missing values
threshold = len(df) * 0.4
df = df.dropna(thresh=threshold, axis=1)

# Dropping rows with any missing values
df = df.dropna()

if 'discoverymethod' in df.columns:
    df['discoverymethod'] = df['discoverymethod'].astype('category').cat.codes

# Reset index
df.reset_index(drop=True, inplace=True)

# Save cleaned data
df.to_csv('cleaned_data.csv', index=False)

print("Cleaned data saved to 'cleaned_data.csv'")
