import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv("data/urlData.csv")
df.dropna(inplace=True)

# Encode Depth feature
def encode_url_depth(threshold):
    return 1 if threshold > 3 else 0

df['Depth'] = df['Depth'].apply(encode_url_depth)

# Drop non-numeric features
X = df.drop(['Label', 'Domain'], axis=1)
y = df['Label']

# Remove HTTPS_Domain if it exists
if 'HTTPS_Domain' in X.columns:
    X = X.drop('HTTPS_Domain', axis=1)

print("=" * 60)
print("PHISHING WEBSITE DETECTION - MODEL ACCURACIES")
print("=" * 60)

# 1. Logistic Regression
print("\n1. LOGISTIC REGRESSION")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=23)
lrc = LogisticRegression(max_iter=50000, random_state=0)
lrc.fit(X_train, y_train)
lr_acc = accuracy_score(y_test, lrc.predict(X_test)) * 100
print(f"   Accuracy: {lr_acc:.2f}%")

# 2. Neural Network
print("\n2. NEURAL NETWORK (MLP)")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)
mlp.fit(X_train_scaled, y_train)
nn_acc = accuracy_score(y_test, mlp.predict(X_test_scaled)) * 100
print(f"   Accuracy: {nn_acc:.2f}%")

# 3. Random Forest
print("\n3. RANDOM FOREST")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_acc = accuracy_score(y_test, rf.predict(X_test)) * 100
print(f"   Accuracy: {rf_acc:.2f}%")

# 4. XGBoost
print("\n4. XGBOOST")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)
xgb_model = xgb.XGBClassifier(learning_rate=0.1, max_depth=7)
xgb_model.fit(X_train, y_train)
xgb_acc = accuracy_score(y_test, xgb_model.predict(X_test)) * 100
print(f"   Accuracy: {xgb_acc:.2f}%")

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
accuracies = {
    'Logistic Regression': lr_acc,
    'Neural Network': nn_acc,
    'Random Forest': rf_acc,
    'XGBoost': xgb_acc
}

for model, acc in sorted(accuracies.items(), key=lambda x: x[1], reverse=True):
    print(f"{model:.<40} {acc:>6.2f}%")

print("=" * 60)
