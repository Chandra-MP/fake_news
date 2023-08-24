import streamlit as st
import pickle 
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from nltk.stem import WordNetLemmatizer

port_stem = PorterStemmer()
wnl = WordNetLemmatizer()

vector_form = pickle.load(open('vector.pkl', 'rb'))
load_model = pickle.load(open('model.pkl', 'rb'))

vector_sms_form = pickle.load(open('sms_vector.pkl', 'rb'))
load_sms_model = pickle.load(open('sms_model.pkl', 'rb'))


def stemming(content):
    con = re.sub('[^a-zA-Z]', ' ', content)
    con = con.lower()
    con = con.split()
    con = [port_stem.stem(word) for word in con if not word in stopwords.words('english')]    
    con = ' '.join(con)
    return con

def text_lemmatizer(content):
    sample_message = re.sub(pattern='[^a-zA-Z]',repl=' ', string = content)
    sample_message = sample_message.lower()
    sample_message_words = sample_message.split()
    sample_message_words = [word for word in sample_message_words if not word in set(stopwords.words('english'))]
    final_message = [wnl.lemmatize(word) for word in sample_message_words]
    final_message = ' '.join(final_message)
    return final_message

def fake_news(news):
    news = stemming(news)
    input_data = [news] 
    vector_form1 = vector_form.transform(input_data)
    model_prediction = load_model.predict(vector_form1)
    return model_prediction

def predict_spam(sample_message):
  final_message = text_lemmatizer(sample_message)
  temp = vector_sms_form.transform([final_message]).toarray()
  model_prediction = load_sms_model.predict(temp)
  return model_prediction



def main():

    st.title("Fake News and Spam Detector")

    choice = st.selectbox("Select an option", [" . . . ", "Fake News Detector", "Spam Email or SMS Detector"], key="choice", help="Choose an option", placeholder="Select . . . ")

    if choice == "Fake News Detector":
        st.subheader("Input the News Content below")
        sentence = st.text_area("Enter the news here", " ", height=200)
        predict_btn = st.button("Predict")

        if predict_btn:
            prediction_class = fake_news(sentence)
            print(prediction_class)
            if prediction_class == [0]:
                st.success("This news is Reliable!")
            elif prediction_class == [1]:
                st.error("This news is Unreliable")
        
    elif choice == "Spam Email or SMS Detector":
        st.subheader("Input the message content below")
        sentence = st.text_area("Go Bonkers!", "", height=200)
        detect_btn = st.button("Detect")

        if detect_btn:
            detection_class = predict_spam(sentence)
            print(detection_class)
            if detection_class == [0]:
                st.success("This message is Authentic!")
            elif detection_class == [1]:
                st.error("This message is a Spam!")


if __name__ == "__main__":
    main()



