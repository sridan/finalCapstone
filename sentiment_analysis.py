#This is a Capstone Project showcasing the NLP preprocessing using Spacy and TextBlob Libraries.
#This Program Performs a Sentiment Analysis on a dataset of product reviews and also 
# performs the Semantics Analysis and returns the compatibility score.

import spacy
import pandas as pd
from textblob import TextBlob
import warnings

# Load the English language model
nlp = spacy.load('en_core_web_sm')

#Load the data into dataframe
df = pd.read_csv('amazon_product_reviews.csv')

#Retrieve the specific column
reviews_data = df['reviews.text']

#check the number of rows
reviews_data.shape

#Drop the columns which  have  null values
clean_data = df.dropna(subset=['reviews.text'])

#Just get the sample data of first selected number of rows

n = int(input('How many sample reviews would you like to include? '))
clean_data = clean_data[:n]

#Creates a Function to remove the stop words

def remove_stopwords(text):
    
    # Process the text using spaCy
    doc = nlp(text)
    # Filter out stopwords
    filtered_text = [token.text for token in doc if not token.is_stop]
    # Join the filtered tokens back into a single string
    filtered_text = ' '.join(filtered_text)
    
    return filtered_text

#Creates a Function to preprocess the text

def preprocess_text(text):
    
    # Convert text to lowercase
    text = text.lower()
    # Strip leading and trailing whitespace
    text = text.strip()
    # Process the text using spaCy
    doc = nlp(text)
    # Filter out stopwords and punctuation
    filtered_text = [token.text for token in doc if not token.is_stop and not token.is_punct]
    # Join the filtered tokens back into a single string
    filtered_text = ' '.join(filtered_text)
    
    return filtered_text

#Creates a Function to analyze the sentiment in the sentence

def analyze_sentiment(text):
    
    # Create a TextBlob object
    blob = TextBlob(text)
    # Get the polarity score
    polarity_score = blob.sentiment.polarity 
    # Classify sentiment based on the polarity score
    if polarity_score > 0:
        sentiment = 'Positive'
    elif polarity_score < 0:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'
        
    return sentiment,polarity_score

#Creates a Function to compare the similarility between two given texts

def compare_similarity(text1, text2):
    
    try:
        
    # Process the texts using spaCy
     doc1 = nlp(text1)
     doc2 = nlp(text2)
     # Suppress UserWarning globally
     warnings.filterwarnings("ignore", category=UserWarning)
   # Calculate the similarity between the texts
     similarity_score = doc1.similarity(doc2)
     
     return similarity_score
     
    except Exception as e:
        
     pass

#Displays the results of Filtered Text,Cleaned Text ,Sentiment Analysis of each sentence and Similarity score between the two texts.
for idx, row in clean_data.iterrows():
    review = row['reviews.text']
    
    #calling the function to remove stopwords
    filtered_text = remove_stopwords(review)
    print("\nFiltered Text:\n", filtered_text)
    
    #calling the function to preprocess the text
    cleaned_text = preprocess_text(review)
    print("\nCleaned Text:\n", cleaned_text)
    
    #calling the function for sentiment analysis
    sentiment = analyze_sentiment(review)
    print(f"\nReview:{sentiment}\n{review}")
    
    #calling the function to compare the similarity scores between the two reviews.
    if idx < len(clean_data) - 1:
        next_review = clean_data.iloc[idx + 1]['reviews.text']
        similarity_score = compare_similarity(review, next_review)
    else:
        next_review = clean_data.iloc[0]['reviews.text']
        similarity_score = compare_similarity(review, next_review)
    
    #Displaying the similarity score  
    print(f"\nSimilarity Score for Review {idx} and Review{idx+1}: {similarity_score}\n")
    print('---------------------------------------------------------------------------')

