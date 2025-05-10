import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data (run this once)
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
try:
    WordNetLemmatizer().lemmatize('running')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('omw-1.4')
except LookupError:
    nltk.download('omw-1.4')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    """
    Preprocesses the input text by removing URLs, mentions, hashtags,
    special characters, converting to lowercase, tokenizing, lemmatizing,
    and removing stop words.

    Args:
        text (str): The text to preprocess.

    Returns:
        str: The preprocessed text.
    """
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Load your dataset
try:
    data = pd.read_csv('App.csv', encoding='latin1')
    data = data[['text', 'sentiment']].dropna()  # Select only 'text' and 'sentiment' and drop NaNs
except FileNotFoundError:
    print("Error: App.csv not found. Make sure it's in the same directory.")
    exit()
except KeyError as e:
    print(f"Error: Required column not found in App.csv.  Check column names.  Error was: {e}")
    exit()
except Exception as e:
    print(f"An unexpected error occurred while loading App.csv: {e}")
    exit()

#  No need to map, sentiment is already 0,1,2.  Check for bad values.
valid_sentiments = [0, 1, 2]
invalid_sentiments = data[~data['sentiment'].isin(valid_sentiments)]
if not invalid_sentiments.empty:
    print("Warning:  Invalid sentiment values found in App.csv.  These rows will be dropped:")
    print(invalid_sentiments)
    data = data[data['sentiment'].isin(valid_sentiments)]
    
if data.empty:
    print("Error: No valid data remaining after filtering.  Please check your App.csv file.")
    exit()
    

data['processed_text'] = data['text'].apply(preprocess_text)

X = data['processed_text']
y = data['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Extraction using TF-IDF
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Initialize individual classifiers
nb_clf = MultinomialNB()
lr_clf = LogisticRegression(solver='liblinear', random_state=42)
svm_clf = LinearSVC(random_state=42)

# Train individual classifiers
try:
    nb_clf.fit(X_train_tfidf, y_train)
    lr_clf.fit(X_train_tfidf, y_train)
    svm_clf.fit(X_train_tfidf, y_train)
except Exception as e:
    print(f"Error training the classifiers: {e}")
    exit()

# Create a VotingClassifier
voting_clf = VotingClassifier(estimators=[('nb', nb_clf), ('lr', lr_clf), ('svm', svm_clf)], voting='hard')
try:
    voting_clf.fit(X_train_tfidf, y_train)
except Exception as e:
    print(f"Error training the VotingClassifier: {e}")
    exit()

# Evaluate the ensemble model
y_pred = voting_clf.predict(X_test_tfidf)
print("Ensemble Accuracy:", accuracy_score(y_test, y_pred))
print("Ensemble Classification Report:\n", classification_report(y_test, y_pred))

# Save the trained ensemble model and the vectorizer
try:
    with open('ensemble_sentiment_model.pkl', 'wb') as model_file:
        pickle.dump(voting_clf, model_file)

    with open('tfidf_vectorizer.pkl', 'wb') as vectorizer_file:
        pickle.dump(tfidf_vectorizer, vectorizer_file)
except Exception as e:
    print(f"Error saving the model or vectorizer: {e}")
    exit()
    

print("Trained ensemble sentiment model and TF-IDF vectorizer saved as ensemble_sentiment_model.pkl and tfidf_vectorizer.pkl.")
