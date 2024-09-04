import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

# Load dataset
categories = ['sci.space', 'rec.autos', 'comp.graphics', 'sci.med']
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
newsgroups_test = fetch_20newsgroups(subset='test', categories=categories)

# Convert to DataFrame
df_train = pd.DataFrame({'text': newsgroups_train.data, 'target': newsgroups_train.target})
df_test = pd.DataFrame({'text': newsgroups_test.data, 'target': newsgroups_test.target})

# TF-IDF Vectorization
stop_words = stopwords.words('english')
vectorizer = TfidfVectorizer(stop_words=stop_words, max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(df_train['text'])
X_test_tfidf = vectorizer.transform(df_test['text'])


from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Initialize the model
model = MultinomialNB()

# Train the model
model.fit(X_train_tfidf, df_train['target'])

# Predictions
predictions = model.predict(X_test_tfidf)

# Evaluation
print("Accuracy:", accuracy_score(df_test['target'], predictions))
print("Classification Report:\n", classification_report(df_test['target'], predictions))


import joblib

# Save the model
joblib.dump(model, 'spam_detector_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')