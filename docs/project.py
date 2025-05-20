import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore

# Generate synthetic data
np.random.seed(100)
data = pd.DataFrame({
    'sensor_1': np.random.normal(50, 10, 500),
    'sensor_2': np.random.normal(30, 5, 500),
    'sensor_3': np.random.normal(100, 20, 500),
    'failure': np.random.choice([0,1], 500, p=[0.9,0.1])
})

# Introduce missing values
data.loc[data.sample(frac=0.05).index, 'sensor_1'] = np.nan

# Summary table before cleaning
print("Summary before cleaning:")
print(data.describe())

# Fill missing values with median
data['sensor_1'] = data['sensor_1'].fillna(data['sensor_1'].median())

# Remove outliers based on z-score
z_scores = np.abs(zscore(data[['sensor_1', 'sensor_2', 'sensor_3']]))
data_clean = data[(z_scores < 3).all(axis=1)]

# Summary table after cleaning
print("\nSummary after cleaning:")
print(data_clean.describe())

# Plot 1: Histogram of sensor_1 before and after cleaning
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
sns.histplot(data['sensor_1'], bins=30, color='skyblue')
plt.title('Sensor 1 Histogram (Before Cleaning)')
plt.subplot(1,2,2)
sns.histplot(data_clean['sensor_1'], bins=30, color='orange')
plt.title('Sensor 1 Histogram (After Cleaning)')
plt.tight_layout()
plt.show()

# Plot 2: Boxplot of sensor_2 (cleaned data)
plt.figure(figsize=(6,4))
sns.boxplot(x=data_clean['sensor_2'], color='lightgreen')
plt.title('Sensor 2 Boxplot (After Cleaning)')
plt.show()

# Plot 3: Correlation heatmap (cleaned data)
plt.figure(figsize=(6,5))
sns.heatmap(data_clean[['sensor_1','sensor_2','sensor_3']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap (After Cleaning)')
plt.show()