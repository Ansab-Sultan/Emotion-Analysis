import streamlit as st
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
import re
import numpy as np
import pickle
from keras.models import Sequential
from keras.layers import Embedding, Dropout, Bidirectional, GRU, BatchNormalization, Dense
from sklearn.model_selection import train_test_split

# Rebuild the model architecture
model = Sequential([
    Embedding(input_dim=50000, output_dim=50, input_shape=(79,)),
    Dropout(0.5),
    Bidirectional(GRU(120, return_sequences=True)),
    Bidirectional(GRU(64, return_sequences=True)),
    BatchNormalization(),
    Bidirectional(GRU(64)),
    Dense(6, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Load the saved weights
model.load_weights('models/model.keras')

# Load the tokenizer
with open('models/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Function to preprocess text
def preprocess_text(text):
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove special characters and punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text)
    # Remove numeric values
    text = re.sub(r'\d+', '', text)
    # Lowercasing
    text = text.lower()
    # Remove stop words
    stop = stopwords.words('english')
    text = ' '.join([word for word in text.split() if word not in (stop)])
    # Remove non-alphanumeric characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

# Function to make predictions
def predict_emotion(text):
    # Preprocess the text
    text = preprocess_text(text)
    # Tokenize the text
    sequences = tokenizer.texts_to_sequences([text])
    # Pad sequences
    maxlen = 79  # Adjust based on the max sequence length used during training
    padded_sequences = pad_sequences(sequences, maxlen=maxlen)
    # Make predictions
    predictions = model.predict(padded_sequences)
    emotion_labels = ['Sadness', 'Joy', 'Love', 'Anger', 'Fear', 'Surprise']
    predicted_emotion = emotion_labels[np.argmax(predictions)]
    return predicted_emotion

# Streamlit app
def main():
    # Title
    st.title('Emotion Detection')

    # Text input
    text_input = st.text_area('Enter your text here:', '')

    # Button to make predictions
    if st.button('Predict'):
        if text_input.strip() == '':
            st.warning('Please enter some text.')
        else:
            predicted_emotion = predict_emotion(text_input)
            st.success(f'Predicted Emotion: {predicted_emotion}')

if __name__ == '__main__':
    main()
