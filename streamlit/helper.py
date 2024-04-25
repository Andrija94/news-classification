import re
import nltk
import joblib
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

def load_models():
    classifier = joblib.load('../model/classifier.pkl')
    vectorizer = joblib.load('../model/vectorizer.pkl')
    stemmer = PorterStemmer()
    return classifier, vectorizer, stemmer


def stemming(stemmer, content):
    content = re.sub('[^a-zA-Z]',' ',content)
    content = content.lower().split()
    stemmed_content = [stemmer.stem(word) for word in content if not word in stopwords.words('english')]
    stemmed_content = " ".join(stemmed_content)
    return stemmed_content

def pred_to_category(pred):
    category_map = {
        0:'business',
        1:'education',
        2:'entertainment',
        3:'sports',
        4:'technology'
     }
    return category_map.get(pred)