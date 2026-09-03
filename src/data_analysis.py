import pandas as pd

# Load the dataset
df = pd.read_csv("data/student-mat.csv", sep=";")

# Display basic information
print("Dataset loaded successfully!")
print("Shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDataset information:")
print(df.info())

print("\nFinal grade statistics:")
print(df["G3"].describe())

print("\nFinal grade distribution:")
print(df["G3"].value_counts().sort_index())

# Create academic risk label
df["risk"] = df["G3"].apply(lambda x: "At Risk" if x < 10 else "Not At Risk")

print("\nRisk distribution:")
print(df["risk"].value_counts())

import matplotlib.pyplot as plt

# Plot risk distribution
risk_counts = df["risk"].value_counts()

plt.figure(figsize=(7, 5))
risk_counts.plot(kind="bar")

plt.title("Student Academic Risk Distribution")
plt.xlabel("Risk Category")
plt.ylabel("Number of Students")
plt.xticks(rotation=0)
plt.tight_layout()

plt.show()