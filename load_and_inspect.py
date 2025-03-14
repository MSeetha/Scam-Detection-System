import pandas as pd

# Load dataset with proper encoding
df = pd.read_csv("socmed_dataset.csv", encoding="ISO-8859-1")

# Display basic info
print(df.info())

# Check for missing values
print("Missing values:\n", df.isnull().sum())

# Display sample rows
print(df.head())


