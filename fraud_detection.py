import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample

# Load the dataset
data = pd.read_csv(r"C:\Users\USER\Downloads\credit_card_fraud_detection\creditcard.csv")

# Inspect the first few rows
print("First 5 rows of the dataset:")
print(data.head())

# Summary of the dataset
print("\nDataset summary:")
print(data.info())

# Check for missing values
print("\nMissing values in each column:")
print(data.isnull().sum())


# Class distribution
class_counts = data['Class'].value_counts()
print("\nClass distribution:")
print(class_counts)

# Plot class distribution
plt.figure(figsize=(6, 4))
sns.barplot(x=class_counts.index, y=class_counts.values, palette="viridis")
plt.title('Class Distribution (0 = Genuine, 1 = Fraudulent)')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()

# Basic statistics for numerical features
print("\nBasic statistics:")
print(data.describe())


# Separate majority and minority classes
df_majority = data[data['Class'] == 0]
df_minority = data[data['Class'] == 1]

# Upsample minority class
df_minority_upsampled = resample(df_minority, 
                                 replace=True,     # Sample with replacement
                                 n_samples=len(df_majority),  # Match the number of majority class samples
                                 random_state=42)  # For reproducibility

# Combine the upsampled minority class with the majority class
data_balanced = pd.concat([df_majority, df_minority_upsampled])

# Check new class distribution
print("\nBalanced class distribution:")
print(data_balanced['Class'].value_counts())
