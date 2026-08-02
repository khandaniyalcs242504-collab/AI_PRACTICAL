import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

print("Libraries imported")

# Load Dataset (Kaggle diabetes.csv)
pima_df = pd.read_csv("diabetes.csv")

print("Dataset Loaded")

# Display first 5 records
print(pima_df.head())

# Predictor variables and target
X = pima_df.drop("Outcome", axis=1)
y = pima_df["Outcome"]

# Standardization
std = StandardScaler()
X = std.fit_transform(X)

print("Transformed Data")
print(X)

# Train-Test Split
test_size = 0.30
seed = 7

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_size,
    random_state=seed
)

# Gaussian Naive Bayes Model
model = GaussianNB()

# Train Model
model.fit(X_train, y_train)

print("\nModel:")
print(model)

# Prediction
predicted = model.predict(X_test)

print("\nPredicted Values:")
print(predicted)

# Confusion Matrix
cm = confusion_matrix(y_test, predicted)
print("\nConfusion Matrix:")
print(cm)

# Accuracy
accuracy = accuracy_score(y_test, predicted)
print("\nAccuracy:", accuracy)

# Model Score
model_score = model.score(X_test, y_test)
print("Model Score:", model_score)

# Prediction Probabilities
y_predictProb = model.predict_proba(X_test)

print("\nPrediction Probabilities:")
print(y_predictProb)

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_predictProb[:, 1])
roc_auc = auc(fpr, tpr)

print("\nROC AUC:", roc_auc)

# Plot ROC Curve
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color='darkorange',
         linewidth=2,
         label='ROC Curve (AUC = %0.2f)' % roc_auc)

plt.plot([0, 1], [0, 1], color='navy', linestyle='--')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.grid(True)

plt.show()
