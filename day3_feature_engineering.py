
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load cleaned data
df = pd.read_csv('titanic_cleaned.csv')

print("="*60)
print("DAY 3: FEATURE ENGINEERING & PREPARATION")
print("="*60)

print("\nBEFORE FEATURE ENGINEERING:")
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# 1. Create new features
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 100], 
                         labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior'])

print("\n1. FEATURE ENGINEERING - Created:")
print(f"  - FamilySize")
print(f"  - IsAlone")
print(f"  - AgeGroup")

# 2. Encode categorical variables
df['Sex_encoded'] = (df['Sex'] == 'male').astype(int)
embarked_dummies = pd.get_dummies(df['Embarked'], prefix='Embarked', drop_first=True)
df = pd.concat([df, embarked_dummies], axis=1)
agegroup_dummies = pd.get_dummies(df['AgeGroup'], prefix='AgeGroup', drop_first=True)
df = pd.concat([df, agegroup_dummies], axis=1)

print("\n2. ENCODING - Converted text to numbers:")
print(f"  - Sex (Male=1, Female=0)")
print(f"  - Embarked (C, Q, S → binary columns)")
print(f"  - AgeGroup (categories → binary columns)")

# 3. Select features for ML
feature_cols = ['Pclass', 'Age', 'Fare', 'SibSp', 'Parch', 'FamilySize', 'IsAlone', 
                'Sex_encoded', 'Embarked_Q', 'Embarked_S',
                'AgeGroup_Teen', 'AgeGroup_Adult', 'AgeGroup_Middle', 'AgeGroup_Senior']

X = df[feature_cols].copy()
y = df['Survived'].copy()

print(f"\n3. FEATURES SELECTED: {len(feature_cols)} features")
print(f"   Shape: {X.shape}")

# 4. Normalize (scale) features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)

print(f"\n4. NORMALIZATION (StandardScaler):")
print(f"   Before: Age mean={X['Age'].mean():.2f}, std={X['Age'].std():.2f}")
print(f"   After:  Age mean={X_scaled_df['Age'].mean():.4f}, std={X_scaled_df['Age'].std():.4f}")

# 5. Save prepared data
X_scaled_df.to_csv('X_scaled.csv', index=False)
y.to_csv('y.csv', index=False)

with open('feature_names.txt', 'w') as f:
    f.write(','.join(feature_cols))

print(f"\n5. DATA SAVED:")
print(f"   - X_scaled.csv (features ready for ML)")
print(f"   - y.csv (target variable)")
print(f"   - feature_names.txt (column names)")

print("\n" + "="*60)
print("DAY 3 COMPLETE - DATA READY FOR ML MODELS!")
print("="*60)
 