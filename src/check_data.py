import pandas as pd

# Load the dataset
df = pd.read_csv("data/student-mat.csv", sep=";")

# Display basic information
print("Dataset Shape:", df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nColumn Names:")
print(df.columns.tolist())

print("\nMissing Values:")
print(df.isnull().sum())    