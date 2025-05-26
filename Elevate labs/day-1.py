import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the dataset
df = pd.read_csv("D:\\Elevate lab\\Titanic-Dataset.csv")

# Display first few rows and dataset info
print("First 5 rows of the dataset:")
print(df.head())
print("\nDataset Info:")
print(df.info())

# 2. Check for missing values
print("\nMissing Values (per column):")
print(df.isnull().sum())

# Summary statistics
print("\nSummary Statistics:")
print(df.describe(include='all'))

# Handle missing values
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

# 3. Fill categorical columns like 'Embarked' with mode
if 'Embarked' in df.columns:
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

print("\nMissing values after handling:")
print(df.isnull().sum())

# Encode categorical variables
# Label encode binary categorical column 'Sex'
if 'Sex' in df.columns:
    le = LabelEncoder()
    df['Sex'] = le.fit_transform(df['Sex'])

# One-hot encode multi-category columns 'Embarked' and 'Pclass'
df = pd.get_dummies(df, columns=['Embarked', 'Pclass'], drop_first=True)

print("\nData after encoding:")
print(df.head())
print("\nData types after encoding:")
print(df.dtypes)

# 4. Feature scaling - Standardize numerical columns
num_cols = df.select_dtypes(include=['float64', 'int64']).columns
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

print("\nStandardized Numerical Features (first 5 rows):")
print(df[num_cols].head())

# 5. Visualize outliers with boxplots
plt.figure(figsize=(15, 8))
for i, col in enumerate(num_cols, 1):
    plt.subplot(2, (len(num_cols)+1)//2, i)
    sns.boxplot(y=df[col])
    plt.title(f'Boxplot of {col}')
plt.tight_layout()
plt.show()

# Remove outliers using IQR method
def remove_outliers(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

df_clean = remove_outliers(df, num_cols)

print(f"\nShape before removing outliers: {df.shape}")
print(f"Shape after removing outliers: {df_clean.shape}")

print("\nData after removing outliers (first 5 rows):")
print(df_clean.head())
