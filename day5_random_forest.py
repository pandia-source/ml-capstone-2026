import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import pickle

print("="*60)
print("DAY 5: RANDOM FOREST CLASSIFIER")
print("="*60)

# Load prepared data
X = pd.read_csv('X_scaled.csv')
y = pd.read_csv('y.csv').values.ravel()

print(f"\nData loaded: {X.shape}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# 1. CREATE & TRAIN RANDOM FOREST
print("\n1. TRAINING RANDOM FOREST")
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
print("  ✓ Model trained!")

# 2. MAKE PREDICTIONS
print("\n2. MAKING PREDICTIONS")
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
y_test_pred_proba = model.predict_proba(X_test)[:, 1]

# 3. EVALUATE
print("\n3. MODEL EVALUATION")
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred)
recall = recall_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)
auc = roc_auc_score(y_test, y_test_pred_proba)

print(f"\n  Accuracy:")
print(f"    Train: {train_accuracy:.4f}")
print(f"    Test:  {test_accuracy:.4f}")
print(f"\n  Precision: {precision:.4f}")
print(f"  Recall: {recall:.4f}")
print(f"  F1-Score: {f1:.4f}")
print(f"  AUC-ROC: {auc:.4f}")

# 4. CROSS-VALIDATION
print("\n4. CROSS-VALIDATION (5-fold)")
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
print(f"  Mean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# 5. FEATURE IMPORTANCE
print("\n5. TOP 10 IMPORTANT FEATURES")
feature_names = X.columns.tolist()
importances = model.feature_importances_
feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values('Importance', ascending=False)

print(feature_importance.head(10).to_string(index=False))

# 6. SAVE
print("\n6. SAVING MODEL")
with open('random_forest_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("  ✓ Model saved!")

# Save results
results = {
    'model_name': 'Random Forest',
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
results_df.to_csv('day5_results.csv', index=False)

print("\n" + "="*60)
print(f"DAY 5 COMPLETE - Test Accuracy: {test_accuracy:.4f}")
print("="*60)

