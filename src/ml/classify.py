import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from collections import Counter


class MLTweetSentiment:
    def __init__(self, train_data_path, inference_data_paths):
        self.train_data_path = train_data_path
        self.inference_data_paths = inference_data_paths

    def import_tweets(self, path, columns, delete_columns):
        dataset = pd.read_csv(path,
                              encoding='latin-1')
        dataset.columns = columns

        for i in delete_columns:
            del dataset[i]

        dataset['text'] = dataset['text'].apply(
                self.preprocess_tweet)

        try:
            dataset.sentiment = dataset.sentiment.replace(4, 1)
        except AttributeError:
            pass

        return dataset

    def preprocess_tweets(self):
        train_dataset_columns = ['sentiment', 'id', 'date',
                                 'flag', 'user', 'text']
        train_dataset_delete_columns = ['flag', 'id', 'user', 'date']
        train_dataset = self.import_tweets(self.train_data_path,
                                           train_dataset_columns,
                                           train_dataset_delete_columns)
        self.inference_dataset_columns = ['date', 'text', 'score']
        inference_dataset_delete_columns = ['date', 'score']
        inference_dataset = []
        for inference_data_path in self.inference_data_paths:
            inference_dataset.append(
                    self.import_tweets(
                        inference_data_path,
                        self.inference_dataset_columns,
                        inference_dataset_delete_columns))

        return train_dataset, inference_dataset

    @staticmethod
    def preprocess_tweet(tweet):
        tweet.lower()
        # convert all urls to string "URL"
        tweet = re.sub(r'((www\.[^\s]+)|(https?://[^\s]+))', 'URL', tweet)
        # convert all @username to "AT_USER"
        tweet = re.sub(r'@[^\s]+', 'AT_USER', tweet)
        # correct all multiple white spaces to a single white space
        tweet = re.sub(r'[\s]+', ' ', tweet)
        # convert "#topic" to just "topic"
        tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
        return tweet

    def convert_to_format_extract_features(self, train_dataset,
                                           inference_datasets,
                                           method, args):
        train_data = np.array(train_dataset.text)
        labels = np.array(train_dataset.sentiment)
        vector_transformer = method(**args)
        vector_transformer.fit(train_data)
        train_features = vector_transformer.transform(train_data)
        inference_features = []
        for inference_dataset in inference_datasets:
            inference_features.append(
                    vector_transformer.transform(inference_dataset.text))
        return train_features, labels, inference_features

    def train_and_evaluate_classifier(self, train_features, labels,
                                      classifier, classifier_args):
        X_train, X_test, y_train, y_test = train_test_split(
                train_features, labels, test_size=0.2, random_state=42)
        model = classifier(**classifier_args)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='macro')
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy, f1, model

    def evaluate_on_brexit(self, model, inference_features):
        for i, inference_feature in enumerate(inference_features):
            dataset = pd.read_csv(inference_dataset_paths[i],
                                  encoding='latin-1')
            dataset.columns = self.inference_dataset_columns
            y_pred = model.predict(inference_feature)
            random_idx = np.random.choice(
                list(range(inference_feature.shape[0])),
                50,
                replace=False).tolist()
            for item in random_idx:
                print(list(dataset.text)[item], y_pred[item])
            print(Counter(y_pred))


if __name__ == '__main__':
    inference_dataset_paths = [
        "../../data/test14-11-2018.csv"]
    #     "../../data/test15-01-2019.csv",
    #     "../../data/test19-06-2017.csv",
    #     "../../data/test22-02-2016.csv",
    #     "../../data/test23-06-2016.csv",
    #     "../../data/test25-11-2018.csv"]

    twitterSentiment = MLTweetSentiment(
            "../../data/twitter_data.csv",
            inference_dataset_paths
            )

    train_dataset, inference_datasets = twitterSentiment.preprocess_tweets()
    tfidf_args = {'stop_words': 'english', 'sublinear_tf': True}
    count_args = {'stop_words': 'english'}
    features_extractors = [TfidfVectorizer, CountVectorizer]
    features_extractors_args = [tfidf_args, count_args]
    classifiers = [LogisticRegression, MultinomialNB, RandomForestClassifier,
                   SVC]
    logistic_args = {'multi_class': 'auto', 'C': 1}
    naive_bayes_args = {}
    random_forest_args = {'n_estimators': [100, 300, 500],
                          'max_depth': [3, 5]}
    svm_args = {}
    classifier_args = [logistic_args, naive_bayes_args, random_forest_args,
                       svm_args]
    for i, feature_extractor in enumerate(features_extractors):
        for j, classifier in enumerate(classifiers):
            try:
                possibilities = list(ParameterGrid(classifier_args[j]))
            except TypeError:
                possibilities = [classifier_args[j]]

            for possibility in possibilities:
                train_features, labels, inference_features = \
                    twitterSentiment.convert_to_format_extract_features(
                                train_dataset, inference_datasets,
                                feature_extractor,
                                features_extractors_args[i])
                print(Counter(labels))
                accuracy, f1, model = \
                    twitterSentiment.train_and_evaluate_classifier(
                        train_features,
                        labels,
                        classifier,
                        possibility
                        )
                print("Feature Extractor: {0}\tModel: {1}\t"
                      "Parameters: {2}\tF1 Score: {3}\tAccuracy:"
                      "{4}".format(feature_extractor, classifier,
                                   classifier_args[j], f1, accuracy))
                twitterSentiment.evaluate_on_brexit(model, inference_features)
