import streamlit as st
import pickle 
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

port_stem = PorterStemmer()
vectorizer = TfidfVectorizer()

vector_form = pickle.load(open('vector.pkl', 'rb'))
load_model = pickle.load(open('model.pkl', 'rb'))

def stemming(content):
    con = re.sub('[^a-zA-Z]', ' ', content)
    con = con.lower()
    con = con.split()
    con = [port_stem.stem(word) for word in con if not word in stopwords.words('english')]    
    con = ' '.join(con)
    return con

def fake_news(news):
    news = stemming(news)
    input_data = [news] 
    vector_form1 = vector_form.transform(input_data)
    model_prediction = load_model.predict(vector_form1)
    return model_prediction

if __name__ == '__main__':
    st.title("Fake News Predictor ")
    st.subheader("Input the News Content below ")
    sentence = st.text_area("Enter the news here, ", "Some news ", height=200)
    predict_btn = st.button("predict")
    if predict_btn:
        prediction_class = fake_news(sentence)
        print(prediction_class)
        if prediction_class == [0]:
            st.success("This news is Reliable!")
        if prediction_class == [1]:
            st.warning("This news is Unreliable")

