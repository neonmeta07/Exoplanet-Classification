import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#Seaborn theme
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

#Cleaned data
df = pd.read_csv("cleaned_data.csv")

if 'koi_disposition' in df.columns:
    plt.figure()
    sns.countplot(data=df, x='koi_disposition')
    plt.title("Exoplanet Disposition Count")
    plt.xlabel("Disposition")
    plt.ylabel("Count")
    plt.show()

#Correlation Heatmap
plt.figure(figsize=(14, 10))
corr = df.select_dtypes(include=['number']).corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

#Histograms of Key Features
key_features = ['koi_period', 'koi_prad', 'koi_teq', 'koi_steff']
for feature in key_features:
    if feature in df.columns:
        plt.figure()
        sns.histplot(df[feature], kde=True, bins=30)
        plt.title(f"Distribution of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()

#Boxplots by Disposition
if 'koi_disposition' in df.columns:
    for feature in key_features:
        if feature in df.columns:
            plt.figure()
            sns.boxplot(data=df, x='koi_disposition', y=feature)
            plt.title(f"{feature} by Disposition")
            plt.xlabel("Disposition")
            plt.ylabel(feature)
            plt.tight_layout()
            plt.show()
