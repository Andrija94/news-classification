import streamlit as st
from helper import *


def main():
    classifier, vectorizer, stemmer = load_models()

    st.title('News Classification')

    input = st.text_area("Enter text: ")

    if st.button('Submit'):
        cleaned_input = stemming(stemmer, input)
        doc = [cleaned_input]
        vectorizer_input = vectorizer.transform(doc)
        pred = classifier.predict(vectorizer_input)
        category = pred_to_category(int(pred[0]))
        st.write('News category is ~~', category, '~~')

if __name__ == '__main__':
    main()