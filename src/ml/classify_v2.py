import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC


def import_tweets(filename, header=None):
    # import data from csv file via pandas library
    tweet_dataset = pd.read_csv(filename, header=header, encoding='latin-1')
    # the column names are based on sentiment140 dataset provided on kaggle
    tweet_dataset.columns = ['sentiment', 'id', 'date', 'flag', 'user', 'text']
    # delete 3 columns: flags, id, user, as they are not required for analysis
    for i in ['flag', 'id', 'user', 'date']:
        del tweet_dataset[i]
    # in sentiment140 dataset, positive = 4, negative = 0;
    # So we change positive to 1
    tweet_dataset.sentiment = tweet_dataset.sentiment.replace(4, 1)
    return tweet_dataset


def preprocess_tweet(tweet):
    # Preprocess the text in a single tweet
    # arguments: tweet = a single tweet in form of string
    # convert the tweet to lower case
    tweet.lower()
    # convert all urls to sting "URL"
    tweet = re.sub(r'((www\.[^\s]+)|(https?://[^\s]+))', 'URL', tweet)
    # convert all @username to "AT_USER"
    tweet = re.sub(r'@[^\s]+', 'AT_USER', tweet)
    # correct all multiple white spaces to a single white space
    tweet = re.sub(r'[\s]+', ' ', tweet)
    # convert "#topic" to just "topic"
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    return tweet


def feature_extraction(data, method="tfidf"):
    # arguments: data = all the tweets in the form of array,
    # method = type of feature extracter
    # methods of feature extractions: "tfidf"
    if method == "tfidf":
        tfv = TfidfVectorizer(sublinear_tf=True,
                              stop_words="english")
        features = tfv.fit_transform(data)
    return features


def train_classifier(features, label, classifier="logistic_regression"):
    # arguments: features = output of feature_extraction(...),
    # label = labels in array form, classifier = type of classifier
    if classifier == "logistic_regression":
        model = LogisticRegression(C=1.)
    elif classifier == "naive_bayes":
        model = MultinomialNB()
    elif classifier == "svm":
        model = SVC()
    else:
        print("Incorrect selection of classifier")
    # fit model to data
    model.fit(features, label)
    # make prediction on the same (train) data
    probability_to_be_positive = model.predict_proba(features)[:, 1]
    # check AUC(Area Under the Roc Curve) to see how well the score
    # discriminates between negative and positive
    print ("auc (train data):", roc_auc_score(label,
                                              probability_to_be_positive))


# apply the preprocess function for all the tweets in the dataset
tweet_dataset = import_tweets("../../data/twitter_data.csv")
print("Imported tweets")
tweet_dataset['text'] = tweet_dataset['text'].apply(preprocess_tweet)
print("Preprocessed tweets")
data = np.array(tweet_dataset.text)
label = np.array(tweet_dataset.sentiment)
features = feature_extraction(data, method="tfidf")
print("Feature extraction")
train_classifier(features, label, "logistic_regression")
print("Trained classifier")
