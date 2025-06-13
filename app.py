import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

#load the LSTM model
model = load_model('next_word.h5')

#load the tokenizer
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)
    

def predict_next_word(model, tokenizer, text, max_sequence_length):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_length:
        token_list = token_list[-(max_sequence_length-1):]
    token_list = pad_sequences([token_list], maxlen=max_sequence_length-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

#streamlit app
st.title("Next Word Prediction App and Early Stopping Callback")

input_text = st.text_input("Enter a sentence:", "To be or not to be")
if st.button("Predict Next Word"):
    max_sequence_length = model.input_shape[1] + 1
    next_word = predict_next_word(model, tokenizer, input_text, max_sequence_length)
    if next_word:
        st.success(f"The next word is: {next_word}")
    else:
        st.error("Could not predict the next word.")
    
    

