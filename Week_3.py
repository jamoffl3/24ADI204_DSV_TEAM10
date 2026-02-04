import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

print("WEEK 3 PRACTICAL : DATA CLEANING SPRINT\n")

print("Loading Datasets...\n")

finance_df = pd.read_csv(
    r"C:\Users\jamvp\Downloads\Bitcoin_12_4_2025-2_5_2026_historical_data_coinmarketcap.csv"
)

tweets_df = pd.read_csv(
    r"C:\Users\jamvp\Downloads\archive (12)\crypto_10k_tweets_(2021_2022Nov).csv",
    engine="python",
    encoding="latin1",
    on_bad_lines="skip"
)

print("Datasets Loaded Successfully\n")

print("Checking Missing Values...\n")

print("Finance Dataset Missing Values:")
print(finance_df.isnull().sum(), "\n")

print("Sentiment Tweets Dataset Missing Values:")
print(tweets_df.isnull().sum(), "\n")

print("Handling Missing Values (Imputation)...\n")

# Convert Date column if present
if "Date" in finance_df.columns:
    finance_df["Date"] = pd.to_datetime(finance_df["Date"], errors="coerce")

# Clean and convert numeric columns
for col in finance_df.columns:
    if finance_df[col].dtype == object:
        finance_df[col] = (
            finance_df[col]
            .str.replace(",", "", regex=True)
            .str.replace("%", "", regex=True)
            .str.replace("$", "", regex=True)
        )
        finance_df[col] = pd.to_numeric(finance_df[col], errors="coerce")

# Select numeric columns ONLY
finance_numeric = finance_df.select_dtypes(include=np.number)

# Drop columns that are fully NaN
finance_numeric = finance_numeric.dropna(axis=1, how="all")

# Mean imputation
finance_df[finance_numeric.columns] = finance_numeric.fillna(
    finance_numeric.mean()
)

print("Missing values in Finance Dataset handled using Mean Imputation\n")

# Handle tweets dataset
if "tweet" in tweets_df.columns:
    tweets_df = tweets_df.dropna(subset=["tweet"])
else:
    tweets_df = tweets_df.dropna()

print("Missing values in Sentiment Dataset handled by Row Deletion\n")

print("Outlier Detection using Boxplot...\n")

if finance_numeric.shape[1] == 0:
    print("No numeric columns available for boxplot.\n")
else:
    plt.figure(figsize=(10, 6))
    finance_numeric.boxplot()
    plt.title("Boxplot for Finance Dataset (Outlier Detection)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

print("Outlier Detection using Z-Score...\n")

z_scores = np.abs(stats.zscore(finance_numeric, nan_policy="omit"))
outliers = (z_scores > 3).any(axis=1)

print("Number of Outlier Rows Detected:", outliers.sum(), "\n")

# Remove outliers
finance_df_cleaned = finance_df.loc[~outliers]

print("Outliers Removed Successfully")
print("Original Finance Dataset Shape:", finance_df.shape)
print("Cleaned Finance Dataset Shape:", finance_df_cleaned.shape, "\n")

print("Week 3 Data Cleaning Completed Successfully.")
