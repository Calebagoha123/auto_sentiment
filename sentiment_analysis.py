import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from wordcloud import WordCloud

nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Initialize VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

def analyze_sentiment(text, analysis_type="Full Text", text_length="Long Form"):
    """
    Performs sentiment analysis on the given text.
    - analysis_type: 'Sentence' for sentence-level or 'Full Text' for overall sentiment
    - text_length: 'Short Form' uses VADER, 'Long Form' uses TextBlob
    """
    # Preprocess text
    
    if text_length == "Short Form":
        # Use VADER for short-form (social media) text
        if analysis_type == "Sentence":
            sentences = sent_tokenize(text)
            results = [sia.polarity_scores(sentence) for sentence in sentences]
            return results
        else:
            return sia.polarity_scores(text)
    
    elif text_length == "Long Form":
        # Use TextBlob for long-form (article-like) text
        if analysis_type == "Sentence":
            sentences = sent_tokenize(text)
            results = [TextBlob(sentence).sentiment for sentence in sentences]
            return [{"polarity": res.polarity, "subjectivity": res.subjectivity} for res in results]
        else:
            blob = TextBlob(text)
            return {"polarity": blob.sentiment.polarity, "subjectivity": blob.sentiment.subjectivity}

def generate_wordcloud(text):
    """
    Generates a word cloud visualization from the input text.
    """
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word.lower() for word in tokens if word.isalnum() and word.lower() not in stop_words]
    
    # Generate the word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(filtered_tokens))
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    return fig

def highlight_sentiment(text):
    """
    Highlights text based on sentiment: green for positive, red for negative, and no highlight for neutral.
    Returns a list of (text, sentiment) tuples.
    """
    sentences = sent_tokenize(text)
    highlighted_text = []
    
    for sentence in sentences:
        sentiment = sia.polarity_scores(sentence)
        if sentiment['compound'] > 0.05:
            highlighted_text.append((sentence, 'positive'))
        elif sentiment['compound'] < -0.05:
            highlighted_text.append((sentence, 'negative'))
        else:
            highlighted_text.append((sentence, 'neutral'))
    
    return highlighted_text

def visualize_sentiment(sentiment_results):
    """
    Visualizes sentiment analysis results.
    For sentence-level analysis, plots bar chart of compound scores (VADER) or polarity (TextBlob) for each sentence.
    For full-text analysis, plots a pie chart for VADER and bar chart for TextBlob.
    """
    if isinstance(sentiment_results, list):
        # Sentence-level analysis visualization
        fig, ax = plt.subplots(figsize=(12, 6))
        
        if 'compound' in sentiment_results[0]:
            # VADER results (Short Form)
            df = pd.DataFrame(sentiment_results)
            df['sentence_index'] = df.index
            sns.barplot(data=df, x='sentence_index', y='compound', ax=ax)
            ax.set_title("Sentence-Level Sentiment Scores (VADER)")
            ax.set_ylabel("Compound Score")
        else:
            # TextBlob results (Long Form)
            df = pd.DataFrame(sentiment_results)
            df['sentence_index'] = df.index
            sns.barplot(data=df, x='sentence_index', y='polarity', ax=ax)
            ax.set_title("Sentence-Level Sentiment Scores (TextBlob)")
            ax.set_ylabel("Polarity Score")
        
        ax.set_xlabel("Sentence Index")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        plt.tight_layout()
    else:
        # Full-text analysis visualization
        if 'compound' in sentiment_results:
            # VADER (Full Text - Short Form)
            labels = ['Positive', 'Neutral', 'Negative']
            sizes = [sentiment_results['pos'], sentiment_results['neu'], sentiment_results['neg']]
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
            ax.set_title("Full Text Sentiment Distribution (VADER)")
        else:
            # TextBlob (Full Text - Long Form)
            fig, ax = plt.subplots(figsize=(8, 6))
            df = pd.DataFrame({'Score Type': ['Polarity', 'Subjectivity'],
                               'Score': [sentiment_results['polarity'], sentiment_results['subjectivity']]})
            sns.barplot(data=df, x='Score Type', y='Score', ax=ax)
            ax.set_title("Full Text Sentiment Scores (TextBlob)")
            ax.set_ylabel("Score")
    
    return fig