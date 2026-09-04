import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

print("Libraries imported")

# Load the restaurant inventory dataset
file_path = "C:/Daniyal khan/AI_PRACTICAL/restaurant_inventory_100days.csv"

dataset = pd.read_csv(file_path)

print("Dataset Loaded")

print("\nFirst 5 rows:")
print(dataset.head())

print("\nDataset Shape:")
print(dataset.shape)

# Select the required columns
dataset = dataset[
    [
        "Current_Stock",
        "Reorder_Level",
        "Daily_Usage",
        "Lead_Time",
        "Price_per_Unit",
        "Seasonal_Factor",
        "Waste_Percentage"
    ]
].copy()

# Convert selected columns to numeric
for column in dataset.columns:
    dataset[column] = pd.to_numeric(
        dataset[column],
        errors="coerce"
    )

# Remove missing values
dataset = dataset.dropna()

print("\nData after preprocessing:")
print(dataset.head())

print("\nDataset Shape after preprocessing:")
print(dataset.shape)

# Create the target variable
# 1 means the item needs to be reordered
# 0 means the item does not need to be reordered
dataset["Reorder_Needed"] = np.where(
    dataset["Current_Stock"] <= dataset["Reorder_Level"],
    1,
    0
)

print("\nReorder Status:")
print(dataset["Reorder_Needed"].value_counts())

# Separate features and target
X = dataset[
    [
        "Current_Stock",
        "Reorder_Level",
        "Daily_Usage",
        "Lead_Time",
        "Price_per_Unit",
        "Seasonal_Factor",
        "Waste_Percentage"
    ]
]

y = dataset["Reorder_Needed"]

print("\nFeatures:")
print(X.head())

print("\nTarget:")
print(y.head())

# Split the dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Data Size:")
print(X_train.shape)

print("\nTesting Data Size:")
print(X_test.shape)

# Create a decision tree with depth 1 as the weak classifier
weak_classifier = DecisionTreeClassifier(
    max_depth=1,
    random_state=42
)

# Train the weak classifier
weak_classifier.fit(
    X_train,
    y_train
)

# Predict using the weak classifier
weak_pred = weak_classifier.predict(X_test)

# Calculate weak classifier accuracy
weak_accuracy = accuracy_score(
    y_test,
    weak_pred
)

print("\nWeak Classifier Accuracy:")
print(
    weak_accuracy * 100,
    "%"
)

# Display weak classifier confusion matrix
print("\nWeak Classifier Confusion Matrix:")
print(
    confusion_matrix(
        y_test,
        weak_pred
    )
)

# Display weak classifier classification report
print("\nWeak Classifier Classification Report:")
print(
    classification_report(
        y_test,
        weak_pred,
        zero_division=0
    )
)

# Create the AdaBoost model
adaboost_model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(
        max_depth=1,
        random_state=42
    ),
    n_estimators=50,
    learning_rate=1.0,
    random_state=42
)

# Train the AdaBoost model
adaboost_model.fit(
    X_train,
    y_train
)

# Predict using AdaBoost
ada_pred = adaboost_model.predict(X_test)

# Calculate AdaBoost accuracy
ada_accuracy = accuracy_score(
    y_test,
    ada_pred
)

print("\nAdaBoost Accuracy:")
print(
    ada_accuracy * 100,
    "%"
)

# Display predicted and actual values
print("\nAdaBoost Predicted Results:")
print(ada_pred)

print("\nActual Results:")
print(y_test.values)

# Display AdaBoost confusion matrix
ada_cm = confusion_matrix(
    y_test,
    ada_pred
)

print("\nAdaBoost Confusion Matrix:")
print(ada_cm)

# Display AdaBoost classification report
print("\nAdaBoost Classification Report:")
print(
    classification_report(
        y_test,
        ada_pred,
        zero_division=0
    )
)

# Compare the two models
improvement = (
    ada_accuracy - weak_accuracy
) * 100

print("\nModel Comparison:")

print(
    "Weak Classifier Accuracy:",
    weak_accuracy * 100,
    "%"
)

print(
    "AdaBoost Accuracy:",
    ada_accuracy * 100,
    "%"
)

print(
    "Improvement using AdaBoost:",
    improvement,
    "percentage points"
)

# Create a graph comparing model accuracy
models = [
    "Weak Classifier",
    "AdaBoost"
]

accuracies = [
    weak_accuracy * 100,
    ada_accuracy * 100
]

plt.figure(figsize=(8, 5))

bars = plt.bar(
    models,
    accuracies
)

plt.xlabel("Model")
plt.ylabel("Accuracy (%)")
plt.title("Weak Classifier vs AdaBoost")
plt.ylim(0, 100)

# Display accuracy values above the bars
for bar, value in zip(
    bars,
    accuracies
):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 1,
        f"{value:.2f}%",
        ha="center"
    )

plt.tight_layout()
plt.show()

# Calculate feature importance
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": adaboost_model.feature_importances_
})

# Sort features according to importance
feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nAdaBoost Feature Importance:")
print(feature_importance)

# Create a graph showing feature importance
plt.figure(figsize=(9, 5))

bars = plt.bar(
    feature_importance["Feature"],
    feature_importance["Importance"]
)

plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("AdaBoost Feature Importance")

plt.xticks(rotation=30)

# Display importance values above the bars
for bar, value in zip(
    bars,
    feature_importance["Importance"]
):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.01,
        f"{value:.3f}",
        ha="center"
    )

plt.tight_layout()
plt.show()

print("\nProgram completed successfully")
