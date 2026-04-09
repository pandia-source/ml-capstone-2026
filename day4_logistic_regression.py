import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

print("="*60)
print("DAY 4: LOGISTIC REGRESSION - BASELINE MODEL")
print("="*60)

# Load prepared data
X = pd.read_csv('X_scaled.csv')
y = pd.read_csv('y.csv').values.ravel()

print(f"\nData loaded:")
print(f"  Features (X): {X.shape}")
print(f"  Target (y): {y.shape}")
print(f"  Survival rate: {y.mean():.2%}")

# 1. SPLIT DATA - Train/Test
print("\n1. SPLITTING DATA")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"  Train set: {X_train.shape} ({len(y_train)} samples)")
print(f"  Test set:  {X_test.shape} ({len(y_test)} samples)")
print(f"  Train survival rate: {y_train.mean():.2%}")
print(f"  Test survival rate:  {y_test.mean():.2%}")

# 2. CREATE & TRAIN MODEL
print("\n2. TRAINING LOGISTIC REGRESSION")
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)
print("  ✓ Model trained!")

# 3. MAKE PREDICTIONS
print("\n3. MAKING PREDICTIONS")
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
y_test_pred_proba = model.predict_proba(X_test)[:, 1]

print(f"  Train predictions shape: {y_train_pred.shape}")
print(f"  Test predictions shape: {y_test_pred.shape}")

# 4. EVALUATE MODEL
print("\n4. MODEL EVALUATION METRICS")

# Accuracy
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"\n  Accuracy:")
print(f"    Train: {train_accuracy:.4f}")
print(f"    Test:  {test_accuracy:.4f}")

# Precision, Recall, F1
precision = precision_score(y_test, y_test_pred)
recall = recall_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)
print(f"\n  Precision (of predicted survivors, how many actually survived): {precision:.4f}")
print(f"  Recall (of actual survivors, how many we caught): {recall:.4f}")
print(f"  F1-Score (balance between precision & recall): {f1:.4f}")

# AUC-ROC
auc = roc_auc_score(y_test, y_test_pred_proba)
print(f"\n  AUC-ROC (0-1, higher is better): {auc:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_test_pred)
print(f"\n  Confusion Matrix:")
print(f"    True Negatives (correct no): {cm[0,0]}")
print(f"    False Positives (wrong yes): {cm[0,1]}")
print(f"    False Negatives (wrong no): {cm[1,0]}")
print(f"    True Positives (correct yes): {cm[1,1]}")

# 5. CROSS-VALIDATION
print("\n5. CROSS-VALIDATION (5-fold)")
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
print(f"  CV Scores: {[f'{score:.4f}' for score in cv_scores]}")
print(f"  Mean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# 6. FEATURE IMPORTANCE
print("\n6. TOP FEATURES (by coefficient magnitude)")
feature_names = X.columns.tolist()
coefficients = model.coef_[0]
feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients,
    'Abs_Coefficient': np.abs(coefficients)
}).sort_values('Abs_Coefficient', ascending=False)

print(feature_importance.head(10).to_string(index=False))

# 7. SAVE MODEL & RESULTS
print("\n7. SAVING MODEL & RESULTS")
import pickle
with open('logistic_regression_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("  ✓ Model saved: logistic_regression_model.pkl")

# Save evaluation results
results = {
    'model_name': 'Logistic Regression',
    'train_accuracy': train_accuracy,
    'test_accuracy': test_accuracy,
    'precision': precision,
    'recall': recall,
    'f1': f1,
    'auc_roc': auc,
    'cv_mean': cv_scores.mean(),
    'cv_std': cv_scores.std()
}

results_df = pd.DataFrame([results])
results_df.to_csv('day4_results.csv', index=False)
print("  ✓ Results saved: day4_results.csv")

print("\n" + "="*60)
print("DAY 4 COMPLETE - BASELINE MODEL TRAINED!")
print("="*60)
print(f"\n✓ Test Accuracy: {test_accuracy:.4f}")
print(f"✓ AUC-ROC: {auc:.4f}")
