import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('shoppingbehaviour.csv')

print(df.head())
print(df.info())
print(df.describe(include='all'))

print("\nMissing values:\n", df.isnull().sum())

plt.figure(figsize=(8,5))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

df.select_dtypes(include=['int64', 'float64']).hist(figsize=(10,8), bins=20, edgecolor='black')
plt.suptitle('Distribution of Numerical Columns')
plt.show()


for col in df.select_dtypes(include='object').columns[:3]:
    plt.figure(figsize=(6,4))
    sns.countplot(x=col, data=df, palette='viridis')
    plt.title(f'Distribution of {col}')
    plt.xticks(rotation=45)
    plt.show()

if 'Age' in df.columns and 'Spending Score (1-100)' in df.columns:
    sns.scatterplot(data=df, x='Age', y='Spending Score (1-100)', hue='Gender')
    plt.title('Age vs Spending Score')
    plt.show()
