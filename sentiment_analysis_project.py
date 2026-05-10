# ==========================================================
# CODEALPHA SENTIMENT ANALYSIS PROJECT
# Project: Amazon Reviews Sentiment Analysis
# ==========================================================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# NLP Libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob

# Download NLTK Data
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Visualization Style
sns.set(style='whitegrid')
plt.rcParams['figure.figsize'] = (10,6)

# ==========================================================
# LOAD DATASET
# ==========================================================

print("Loading dataset...")

try:
    df = pd.read_csv('amazon_reviews.csv')
    print("Dataset loaded successfully!\n")
except FileNotFoundError:
    print("Dataset file not found.")
    exit()

# ==========================================================
# BASIC INFORMATION
# ==========================================================

print("========== FIRST 5 ROWS ==========")
print(df.head())

print("\n========== DATASET SHAPE ==========")
print(df.shape)

print("\n========== COLUMN NAMES ==========")
print(df.columns)

# ==========================================================
# SELECT REVIEW COLUMN
# ==========================================================

# Change column name if needed

review_column = 'reviews.text'

# Check if column exists

if review_column not in df.columns:
    print(f"Column '{review_column}' not found.")
    exit()

# Keep only required data

reviews_df = df[[review_column]].copy()

# Remove missing reviews

reviews_df.dropna(inplace=True)

# Rename column

reviews_df.rename(columns={review_column: 'Review'}, inplace=True)

print("\nTotal Reviews:", len(reviews_df))

# ==========================================================
# TEXT CLEANING FUNCTION
# ==========================================================

stop_words = set(stopwords.words('english'))


def clean_text(text):

    # Convert to lowercase
    text = text.lower()

    # Tokenization
    words = word_tokenize(text)

    # Remove non-alphabetic words
    words = [word for word in words if word.isalpha()]

    # Remove stopwords
    words = [word for word in words if word not in stop_words]

    # Join words
    return ' '.join(words)

# Apply cleaning

print("\nCleaning reviews...")

reviews_df['Cleaned_Review'] = reviews_df['Review'].apply(clean_text)

print("Text cleaning completed!\n")

# ==========================================================
# SENTIMENT ANALYSIS
# ==========================================================


def get_sentiment(text):

    analysis = TextBlob(text)

    polarity = analysis.sentiment.polarity

    if polarity > 0:
        return 'Positive'

    elif polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

# Apply sentiment analysis

print("Performing sentiment analysis...")

reviews_df['Sentiment'] = reviews_df['Cleaned_Review'].apply(get_sentiment)

print("Sentiment analysis completed!\n")

# ==========================================================
# SENTIMENT DISTRIBUTION
# ==========================================================

sentiment_counts = reviews_df['Sentiment'].value_counts()

print("========== SENTIMENT DISTRIBUTION ==========")
print(sentiment_counts)

# Bar Chart

plt.figure(figsize=(8,5))
sns.countplot(
    x='Sentiment',
    data=reviews_df
)

plt.title('Sentiment Distribution of Amazon Reviews')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()

# ==========================================================
# PIE CHART VISUALIZATION
# ==========================================================

plt.figure(figsize=(8,8))

plt.pie(
    sentiment_counts.values,
    labels=sentiment_counts.index,
    autopct='%1.1f%%'
)

plt.title('Sentiment Percentage Distribution')
plt.show()
# ==========================================================
# WORD COUNT ANALYSIS
# ==========================================================

reviews_df['Word_Count'] = reviews_df['Cleaned_Review'].apply(
    lambda x: len(x.split())
)

plt.figure(figsize=(10,6))

sns.histplot(reviews_df['Word_Count'], bins=30)

plt.title('Review Word Count Distribution')
plt.xlabel('Number of Words')
plt.ylabel('Frequency')
plt.show()

# ==========================================================
# SAMPLE REVIEWS
# ==========================================================

print("\n========== SAMPLE REVIEWS ==========")

print(reviews_df[['Review', 'Sentiment']].head(10))

# ==========================================================
# KEY INSIGHTS
# ==========================================================

print("\n========== KEY INSIGHTS ==========")

print("1. Most customer reviews are positive.")
print("2. Negative reviews indicate customer dissatisfaction.")
print("3. Neutral reviews contain balanced opinions.")
print("4. Sentiment analysis helps understand customer behavior.")
print("5. NLP techniques simplify large-scale text analysis.")

# ==========================================================
# EXPORT RESULTS
# ==========================================================

reviews_df.to_csv('amazon_review_sentiments.csv', index=False)

print("\nSentiment results exported successfully!")

# ==========================================================
# PROJECT COMPLETED
# ==========================================================

print("\nSentiment Analysis Project Completed Successfully!")