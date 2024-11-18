import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load dataset with tab separator
df = pd.read_csv("spam.csv", encoding='latin-1', delimiter='\t', on_bad_lines='skip')

# Rename columns for clarity
df.columns = ['label', 'text']

# Map labels to binary values (ham = 0, spam = 1)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Split the data into training and test sets
X = df['text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text to numerical features
tfidf = TfidfVectorizer(stop_words='english', max_features=3000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Evaluate the model
y_pred = model.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Test with custom input
def predict_spam(email):
    email_tfidf = tfidf.transform([email])
    prediction = model.predict(email_tfidf)
    return "Spam" if prediction == 1 else "Not Spam"

# Example usage
test_email = "Congratulations! You've won a free trip to Paris!"
print(f"Prediction for email: '{test_email}' => {predict_spam(test_email)}")

