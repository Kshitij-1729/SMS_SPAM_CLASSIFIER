import streamlit as st
#essentail lib (that are used in code for jupyter notebook)
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

# the function is copied from jupyter
def transform_text(text):
    text=text.lower()  #converting the text into lower case
    
    text=nltk.word_tokenize(text)  #word_tokenize break the text into words and make a list out of it
    
    L=[]
    for i in text:
        if i.isalnum():    # if the word is alphabet or numeric or both then
            L.append(i)

    y=[]
    for i in L:
        if i not in stopwords.words('english') and i not in string.punctuation: # the word sholud not be a stopword and niether a puntuations then
            y.append(i)

    x=[]
    for i in y:
        x.append(ps.stem(i))
            
    return x


cv=pickle.load(open('vectorizer.pkl','rb'))
mdl=pickle.load(open('model.pkl','rb'))

st.title('Email/SMS Spam Classifier')

input_sms=st.text_area("Enter the message: ")

if st.button('Predict'):
    # 1. preprocess
    transformed_sms = transform_text(input_sms)            # likely returns a list of words
    transformed_sms_str = " ".join(transformed_sms)        # join tokens(words) back to string for vectorizer

    # 2. vectorize
    vector_input = cv.transform([transformed_sms_str])     # transforms the list of strings into vectors

    # 3. predict
    result = mdl.predict(vector_input)                      # predict returns an array

    # 4. Display Result
    if result == 1:
        st.header('Spam')
    else:
        st.header('Not Spam')