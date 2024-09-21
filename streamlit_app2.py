import streamlit as st
from textblob import TextBlob

# Title of the app
st.title("Simple Sentiment Analysis App")

# Instructions for the user
st.write("Enter text below to analyze its sentiment (polarity and subjectivity).")

# Text input from the user
text_input = st.text_area("Enter your text:")

# Perform sentiment analysis when the button is clicked
if st.button("Analyze Sentiment"):
    if text_input:
        # Create a TextBlob object
        blob = TextBlob(text_input)
        
        # Get the sentiment polarity and subjectivity
        sentiment_polarity = blob.sentiment.polarity
        sentiment_subjectivity = blob.sentiment.subjectivity
        
        # Display the results
        st.write(f"Sentiment Polarity: {sentiment_polarity}")
        st.write(f"Sentiment Subjectivity: {sentiment_subjectivity}")
    else:
        st.warning("Please enter some text!")

