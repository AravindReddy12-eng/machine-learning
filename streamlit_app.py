import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd

# Title of the web app
#st.title("Text Vectorization App")
st.markdown(
    """
    <style>
    .title {
        text-align: center;
        color: #4CAF50;
        font-size: 30px;
    }
    </style>
    <h1 class="title">Text Vectorization App</h1>
    """, 
    unsafe_allow_html=True
)

# Instructions for the user
st.write("Enter text below and choose vectorization method (Bag of Words, TF-IDF, or N-grams).")

# Text input from the user
text_input = st.text_area("Enter your text (each sentence on a new line):")

# Vectorization method choice
vectorization_method = st.selectbox(
    "Select a vectorization method",
    ("Bag of Words", "TF-IDF", "N-grams")
)
st.markdown(
    """
    <style>
    .radio-label {
        font-size: 16px;
        margin-right: 15px;
    }
    .bag-of-words {
        color: #ff6347; /* Tomato Red */
    }
    .tf-idf {
        color: #4CAF50; /* Green */
    }
    .ngrams {
        color: #1E90FF; /* Dodger Blue */
    }
    </style>
    """, 
    unsafe_allow_html=True
)

# Custom radio buttons using st.radio for selecting vectorization method
vectorization_method = st.radio(
    "Select a vectorization method",
    ("Bag of Words", "TF-IDF", "N-grams"),
    format_func=lambda x: f"<span class='{x.lower().replace(' ', '-')}'>{x}</span>",
    key="vectorization_method"
)

# Map the selected option to a CSS class and render with colors
if vectorization_method == "Bag of Words":
    st.markdown("<p class='radio-label bag-of-words'>Bag of Words selected</p>", unsafe_allow_html=True)
elif vectorization_method == "TF-IDF":
    st.markdown("<p class='radio-label tf-idf'>TF-IDF selected</p>", unsafe_allow_html=True)
elif vectorization_method == "N-grams":
    st.markdown("<p class='radio-label ngrams'>N-grams selected</p>", unsafe_allow_html=True)


# N-gram range selection (if N-grams is chosen)
ngram_range = (1, 1)  # default to unigrams
if vectorization_method == "N-grams":
    ngram_min = st.slider("Select the minimum n-gram range", 1, 3, 1)
    ngram_max = st.slider("Select the maximum n-gram range", 1, 3, 2)
    ngram_range = (ngram_min, ngram_max)

# Perform vectorization when the button is clicked
if st.button("Convert Text to Vectors"):
    if text_input:
        corpus = text_input.split("\n")  # split input into sentences

        # Bag of Words
        if vectorization_method == "Bag of Words":
            vectorizer = CountVectorizer()
        # TF-IDF
        elif vectorization_method == "TF-IDF":
            vectorizer = TfidfVectorizer()
        # N-grams
        elif vectorization_method == "N-grams":
            vectorizer = CountVectorizer(ngram_range=ngram_range)

        # Fit and transform the corpus
        X = vectorizer.fit_transform(corpus)

        # Convert to DataFrame for better visualization
        df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
        st.write("Vectorized Text Data:")
        st.dataframe(df)
    else:
        st.warning("Please enter some text!")
