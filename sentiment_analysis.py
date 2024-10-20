import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

# Load Dataset

# URL of the raw CSV file from GitHub
url = 'https://raw.githubusercontent.com/AravindReddy12-eng/machine-learning/main/IMDB%20Dataset.csv'

# Load the dataset
df = pd.read_csv(url)

# Continue with preprocessing and model setup...


# Preprocess Data
stop_words = set(stopwords.words('english'))
def preprocess_text(text):
    text = text.lower().split()
    text = [word for word in text if word.isalnum() and word not in stop_words]
    return ' '.join(text)

df['cleaned_review'] = df['review'].apply(preprocess_text)

# Tokenize and Pad Sequences
max_words = 10000
max_len = 100
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(df['cleaned_review'])

X = tokenizer.texts_to_sequences(df['cleaned_review'])
X = pad_sequences(X, maxlen=max_len)

# Encode Target Labels
y = pd.get_dummies(df['sentiment']).values

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build LSTM Model
model = Sequential()
model.add(Embedding(max_words, 128, input_length=max_len))
model.add(LSTM(128, return_sequences=False))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

# Compile Model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train Model (For quick demo, use fewer epochs, increase for better results)
model.fit(X_train, y_train, epochs=2, batch_size=64, validation_split=0.2)

# Save model for later use
model.save('sentiment_lstm_model.h5')

# Streamlit Interface
st.title("Sentiment Analysis on Movie Reviews")
st.write("Enter a movie review and get the predicted sentiment (Positive or Negative).")

# Input review from user
user_review = st.text_area("Enter your movie review:")

# Preprocess the input review
if user_review:
    def preprocess_input_text(text):
        text = text.lower().split()
        text = [word for word in text if word.isalnum() and word not in stop_words]
        return ' '.join(text)
    
    cleaned_review = preprocess_input_text(user_review)
    
    # Tokenize and pad the input review
    review_seq = tokenizer.texts_to_sequences([cleaned_review])
    review_padded = pad_sequences(review_seq, maxlen=max_len)

    # Load trained model
    model = load_model('sentiment_lstm_model.h5')
    
    # Predict sentiment
    prediction = model.predict(review_padded)
    sentiment = np.argmax(prediction, axis=1)
    
    if sentiment == 1:
        st.write("**Prediction: Positive Sentiment** 😀")
    else:
        st.write("**Prediction: Negative Sentiment** 😔")
