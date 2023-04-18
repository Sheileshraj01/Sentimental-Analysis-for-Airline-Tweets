import numpy as np 
import pandas as pd 
import re
import nltk
import sklearn


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

#Random Forest classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(processed_features, labels, test_size=0.2, random_state=0)
text_classifier = RandomForestClassifier(n_estimators=100, random_state=0)
text_classifier.fit(x_train, y_train)
predictions = text_classifier.predict(x_test)
from sklearn.metrics import accuracy_score
print('accuracy score for random forest', accuracy_score(y_test, predictions)) 

#Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(processed_features, labels, test_size=0.2)
lr = LogisticRegression()
lr.fit(x_train, y_train)
lr.fit(x_test, y_test)
predictions = lr.predict(x_test)
print('accuracy score for logistic regression', accuracy_score(y_test, predictions)) 


#Naive Bayes classifier
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(x_train, y_train)
model.fit(x_test, y_test)
predictions= model.predict(x_test)
from sklearn.metrics import accuracy_score
print('accuracy score for naive bayes classifier', accuracy_score(predictions, y_test))

#Support Vector Machine classifier
from sklearn.svm import SVC
svm = SVC()
svm.fit(x_train, y_train)
predictions = svm.predict(x_test)
from sklearn.metrics import accuracy_score
print('accuracy score for support vector machine', accuracy_score(y_test, predictions))
