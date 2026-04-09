import pandas as pd
import numpy as np

# Load data
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
df = pd.read_csv(url)

print("BEFORE CLEANING:")
print(f"Shape: {df.shape}")
print(f"Missing values:\n{df.isnull().sum()}")

# 1. Handle missing Age (fill with median)
df['Age'].fillna(df['Age'].median(), inplace=True)

# 2. Handle missing Embarked (fill with mode)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# 3. Drop Cabin (too many missing)
df.drop('Cabin', axis=1, inplace=True)

# 4. Drop PassengerId, Name, Ticket (not useful for ML)
df.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)

print("\nAFTER CLEANING:")
print(f"Shape: {df.shape}")
print(f"Missing values:\n{df.isnull().sum()}")
print(f"\nCleaned dataset head:\n{df.head()}")

# Save cleaned data
df.to_csv('titanic_cleaned.csv', index=False)
print("\nCleaned data saved to titanic_cleaned.csv")
