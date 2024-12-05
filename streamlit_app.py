import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load dataset
@st.cache
def load_data():
    df = pd.read_csv("spam.csv", encoding='latin-1', delimiter='\t', on_bad_lines='skip')
    df.columns = ['label', 'text']
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    return df

# Load data
df = load_data()

# Train the model
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
clf = LogisticRegression()
clf.fit(X_train_vec, y_train)

# Streamlit App
st.title("Spam Email Classifier")
st.write("Enter an email to see if it's spam or not.")

user_input = st.text_area("Email content:")
if st.button("Classify"):
    if user_input:
        user_input_vec = vectorizer.transform([user_input])
        prediction = clf.predict(user_input_vec)
        st.write("Prediction: **Spam**" if prediction == 1 else "Prediction: **Ham**")
    else:
        st.write("Please enter text to classify.")
