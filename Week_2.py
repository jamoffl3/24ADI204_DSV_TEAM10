import pandas as pd
import numpy as np

print("WEEK 2 PRACTICAL : DATA LOADING AND INSPECTION\n")

print("Loading Data...")

finance_df = pd.read_csv(
    r"C:\Users\jamvp\Downloads\Bitcoin_12_4_2025-2_5_2026_historical_data_coinmarketcap.csv")
tweets_df = pd.read_csv(
    r"C:\Users\jamvp\Downloads\archive (12)\crypto_10k_tweets_(2021_2022Nov).csv",
    engine="python",
    encoding="latin1",
    on_bad_lines="skip")

print("Datasets Loaded Successfully\n")

print("Inspecting Dataset Structure...\n")

print("Finance Dataset Shape:")
print(finance_df.shape, "\n")

print("Sentiment Tweets Dataset Shape:")
print(tweets_df.shape, "\n")

print("Finance Dataset Columns:")
print(finance_df.columns, "\n")

print("Sentiment Tweets Dataset Columns:")
print(tweets_df.columns, "\n")

print("Previewing Finance Dataset:")
print(finance_df.head(), "\n")

print("Previewing Sentiment Tweets Dataset:")
print(tweets_df.head(), "\n")

print("Checking Data Types...\n")

print("Finance Dataset Data Types:")
print(finance_df.dtypes, "\n")

print("Sentiment Tweets Dataset Data Types:")
print(tweets_df.dtypes, "\n")

print("Week 2 Data Inspection Completed Successfully.")
