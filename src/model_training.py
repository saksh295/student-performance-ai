import pandas as pd

# Load dataset
df = pd.read_csv("data/student-mat.csv", sep=";")

# Create risk label
df["risk"] = df["G3"].apply(
    lambda x: "At Risk" if x < 10 else "Not At Risk"
)

# Select features
features = [
    "age",
    "Medu",
    "Fedu",
    "traveltime",
    "studytime",
    "failures",
    "schoolsup",
    "famsup",
    "paid",
    "activities",
    "higher",
    "internet",
    "freetime",
    "goout",
    "health",
    "absences",
    "G1",
    "G2"
]

X = df[features]
y = df["risk"]

X = pd.get_dummies(X, drop_first=True)

print("Features selected:", len(features))
print("X shape:", X.shape)
print("Target shape:", y.shape)

print("\nTarget distribution:")
print(y.value_counts())

from sklearn.model_selection import train_test_split

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Train Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(f"{accuracy:.2%}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

import joblib

# Save the trained model
joblib.dump(model, "models/student_risk_model.pkl")

print("\nModel saved successfully!")