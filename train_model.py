import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
import os

# Create a sample dataset if spam.csv doesn't exist
# This is just to ensure the script runs and generates the pkl files for demonstration
data_exists = False
if os.path.exists("spam.csv"):
    df = pd.read_csv("spam.csv", encoding="latin-1")
    data_exists = True
else:
    print("Warning: spam.csv not found. creating dummy data for demonstration.")
    data = {
        'v1': ['ham', 'spam', 'ham', 'spam', 'ham'],
        'v2': [
            'Hello there, how are you?',
            'Win a free iPhone now!',
            'Meeting at 3pm',
            'Congratulations, you won the lottery!',
            'Can we grab lunch?'
        ]
    }
    df = pd.DataFrame(data)

# Preprocessing (Simplified)
# Assuming v1 is target and v2 is text
X = df['v2']
y = df['v1']

# 1. Initialize Vectorizer
tfidf = TfidfVectorizer(stop_words='english', max_features=3000)

# 2. FIT the vectorizer! This is the missing step in your error.
X_transformed = tfidf.fit_transform(X)

# 3. Train Model
model = MultinomialNB()
model.fit(X_transformed, y)

# 4. Save both properly
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Success: vectorizer.pkl and model.pkl have been saved.")
print("The vectorizer is now fitted.")
