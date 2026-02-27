import pandas as pd

crypto_df = pd.read_csv("crypto_prices.csv")
sentiment_df = pd.read_csv("sentiment_data.csv")

print("=== Crypto Dataset Preview ===")
print(crypto_df.head())

print("\n=== Sentiment Dataset Preview ===")
print(sentiment_df.head())

print("\n=== Crypto Dataset Info ===")
crypto_df.info()

print("\n=== Sentiment Dataset Info ===")
sentiment_df.info()

print("\n=== Crypto Dataset Description ===")
print(crypto_df.describe())

print("\n=== Sentiment Dataset Description ===")
print(sentiment_df.describe(include='all'))

print("\n=== Missing Values in Crypto Dataset ===")
print(crypto_df.isnull().sum())

print("\n=== Missing Values in Sentiment Dataset ===")
print(sentiment_df.isnull().sum())

print("\nDuplicate rows in Crypto:", crypto_df.duplicated().sum())
print("Duplicate rows in Sentiment:", sentiment_df.duplicated().sum())

print("\nCrypto Columns:", crypto_df.columns.tolist())
print("Sentiment Columns:", sentiment_df.columns.tolist())

print("\nCrypto Shape:", crypto_df.shape)
print("Sentiment Shape:", sentiment_df.shape)
