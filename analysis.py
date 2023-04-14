import numpy as np 
import pandas as pd 
import re
import nltk
import sklearn
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

airline_tweets = pd.read_csv('Tweets.csv')
airline_tweets.head()

features = airline_tweets.iloc[:, 10].values
labels = airline_tweets.iloc[:, 1].values

processed_features = []

for sentence in range(0, len(features)):
    # Remove all the special characters
    processed_feature = re.sub(r'\W', ' ', str(features[sentence]))

    # remove all single characters
    processed_feature= re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_feature)

    # Remove single characters from the start
    processed_feature = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_feature) 

    # Substituting multiple spaces with single space
    processed_feature = re.sub(r'\s+', ' ', processed_feature, flags=re.I)

    # Removing prefixed 'b'
    processed_feature = re.sub(r'^b\s+', '', processed_feature)

    # Converting to Lowercase
    processed_feature = processed_feature.lower()

    processed_features.append(processed_feature)

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer (max_features=2500, min_df=7, max_df=0.8, stop_words=stopwords.words('english'))
processed_features = vectorizer.fit_transform(processed_features).toarray()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(processed_features, labels, test_size=0.2, random_state=0)

from sklearn.ensemble import RandomForestClassifier
text_classifier = RandomForestClassifier(n_estimators=200, random_state=0)
text_classifier.fit(X_train, y_train)
rf_predictions = text_classifier.predict(X_test)

nb_classifier = MultinomialNB()
lr_classifier = LogisticRegression()
svm_classifier = SVC(kernel='linear')

nb_classifier.fit(X_train, y_train)
nb_predictions = nb_classifier.predict(X_test)

lr_classifier.fit(X_train, y_train)
lr_predictions = lr_classifier.predict(X_test)

svm_classifier.fit(X_train, y_train)
svm_predictions = svm_classifier.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score

print("Random Forest Classifier:")
print(confusion_matrix(y_test, rf_predictions))
print('Accuracy score:', accuracy_score(y_test, rf_predictions))

print("\nNaive Bayes Classifier:")
print(confusion_matrix(y_test, nb_predictions))
print('Accuracy score:', accuracy_score(y_test, nb_predictions))

print("\nLogistic Regression:")
print(confusion_matrix(y_test, lr_predictions))
print('Accuracy score:', accuracy_score(y_test, lr_predictions))

print("\nSupport Vector Machine:")
print(confusion_matrix(y_test, svm_predictions))
print('Accuracy score:', accuracy_score(y_test, svm_predictions))