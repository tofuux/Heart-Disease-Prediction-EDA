# Heart Disease EDA Program

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# Load Dataset
# -------------------------------
df = pd.read_csv("Heart_Disease_Prediction.csv")   # change name ONLY if your csv name is different

# -------------------------------
# Basic Data Information
# -------------------------------
print("Shape of data:", df.shape)
print("\nFirst 5 records:\n", df.head())
print("\nStatistical Summary:\n", df.describe())
print("\nMissing Values:\n", df.isnull().sum())

# -------------------------------
# Correlation Heatmap (Numeric only)
# -------------------------------
plt.figure(figsize=(10,8))
sns.heatmap(
    df.select_dtypes(include='number').corr(),
    annot=True,
    cmap="coolwarm"
)
plt.title("Correlation Matrix")
plt.show()

# -------------------------------
# Heart Disease Count Plot
# -------------------------------
sns.countplot(x="Heart Disease", data=df)
plt.title("Heart Disease Distribution")
plt.xlabel("Heart Disease Status")
plt.ylabel("Count")
plt.show()

# -------------------------------
# Age Distribution
# -------------------------------
sns.histplot(df["Age"], kde=True)
plt.title("Age Distribution of Patients")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

# -------------------------------
# Cholesterol vs Heart Disease
# -------------------------------
sns.boxplot(x="Heart Disease", y="Cholesterol", data=df)
plt.title("Cholesterol Levels vs Heart Disease")
plt.show()
