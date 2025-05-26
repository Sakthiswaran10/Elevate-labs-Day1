# Titanic Dataset Preprocessing and Visualization

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the dataset
df = pd.read_csv("D:/Elevate labs/DAY 1/Titanic-Dataset.csv")

# Display initial data overview
print("First 5 rows of the dataset:")
print(df.head())
print("\nDataset Info:")
df.info()

# 2. Check for missing values
print("\nMissing Values (per column):")
print(df.isnull().sum())

# Display summary statistics
print("\nSummary Statistics:")
print(df.describe(include='all'))

# 3. Handle missing values
# Fill 'Age' with mean
if 'Age' in df.columns:
    df['Age'] = df['Age'].fillna(df['Age'].mean())

# Fill 'Fare' with median
if 'Fare' in df.columns:
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())

# Impute remaining numeric columns with mean
num_cols = df.select_dtypes(include=['float64', 'int64']).columns
imputer = SimpleImputer(strategy='mean')
df[num_cols] = imputer.fit_transform(df[num_cols])

# Fill 'Embarked' with mode
if 'Embarked' in df.columns:
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

print("\nMissing values after handling:")
print(df.isnull().sum())

# 4. Encode categorical variables
# Label encode binary column 'Sex'
if 'Sex' in df.columns:
    le = LabelEncoder()
    df['Sex'] = le.fit_transform(df['Sex'])

# One-hot encode 'Embarked' and 'Pclass'
df = pd.get_dummies(df, columns=['Embarked', 'Pclass'], drop_first=True)

print("\nData after encoding:")
print(df.head())
print("\nData types after encoding:")
print(df.dtypes)

# 5. Feature scaling - Standardize numerical columns
num_cols = df.select_dtypes(include=['float64', 'int64']).columns
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

print("\nStandardized Numerical Features (first 5 rows):")
print(df[num_cols].head())

# 6. Visualize outliers using boxplots
plt.figure(figsize=(15, 8))
for i, col in enumerate(num_cols, 1):
    plt.subplot(2, (len(num_cols) + 1) // 2, i)
    sns.boxplot(y=df[col])
    plt.title(f'Boxplot of {col}')
plt.tight_layout()
plt.show()

# 7. Remove outliers using IQR method
def remove_outliers(data, columns):
    for col in columns:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        data = data[(data[col] >= lower) & (data[col] <= upper)]
    return data

df_clean = remove_outliers(df, num_cols)

print(f"\nShape before removing outliers: {df.shape}")
print(f"Shape after removing outliers: {df_clean.shape}")
print("\nCleaned Data Preview (first 5 rows):")
print(df_clean.head())
