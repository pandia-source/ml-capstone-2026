
import pandas as pd
import numpy as np

# Load Titanic dataset
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
df = pd.read_csv(url)

print("="*60)
print("TITANIC DATASET - DETAILED EDA")
print("="*60)

# 1. Dataset Overview
print("\n1. DATASET OVERVIEW")
print(f"Shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nData types:\n{df.dtypes}")

# 2. Missing Values
print("\n2. MISSING VALUES")
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing_Count': missing,
    'Percentage': missing_pct
})
print(missing_df[missing_df['Missing_Count'] > 0])

# 3. Basic Statistics
print("\n3. BASIC STATISTICS")
print(df.describe())

# 4. Categorical Columns
print("\n4. CATEGORICAL COLUMNS")
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    print(f"\n{col}:")
    print(df[col].value_counts())

# 5. Target Variable (Survived)
print("\n5. TARGET VARIABLE - SURVIVED")
print(df['Survived'].value_counts())
print(f"\nSurvival Rate: {df['Survived'].mean():.2%}")

# 6. Correlation with Survival
print("\n6. CORRELATION WITH SURVIVAL")
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if col != 'Survived':
        corr = df[col].corr(df['Survived'])
        print(f"{col}: {corr:.3f}")

# 7. Age Analysis
print("\n7. AGE ANALYSIS")
print(f"Age - Mean: {df['Age'].mean():.2f}, Median: {df['Age'].median():.2f}, Std: {df['Age'].std():.2f}")
print(f"Age - Min: {df['Age'].min()}, Max: {df['Age'].max()}")

# 8. Fare Analysis
print("\n8. FARE ANALYSIS")
print(f"Fare - Mean: {df['Fare'].mean():.2f}, Median: {df['Fare'].median():.2f}, Std: {df['Fare'].std():.2f}")
print(f"Fare - Min: {df['Fare'].min()}, Max: {df['Fare'].max()}")

# 9. Class Analysis
print("\n9. PASSENGER CLASS ANALYSIS")
print(df['Pclass'].value_counts().sort_index())
print("\nSurvival by Class:")
print(df.groupby('Pclass')['Survived'].mean())

# 10. Gender Analysis
print("\n10. GENDER ANALYSIS")
print(df['Sex'].value_counts())
print("\nSurvival by Gender:")
print(df.groupby('Sex')['Survived'].mean())

print("\n" + "="*60)
print("EDA COMPLETE")
print("="*60)
