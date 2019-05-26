import numpy as np
import pandas as pd
import re
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from collections import Counter
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


class MLTweetSentiment:
    """A class that takes a file path and trains a model performing sentiment
    analysis on tweets (classifies a tweet as positive or negative).

    Attributes:
        inference_dataset_columns (list): The columns that have to be deleted
            from the pandas frame of the inference sets.

    """
    def __init__(self, train_data_path, inference_data_paths):
        """The constructor of the class.

        Args:
            train_data_path (str): A string that contains the path for the
                training set.
            inference_data_paths (list): A list of strings that contains the
                paths for the inference sets.
        """
        self.train_data_path = train_data_path
        self.inference_data_paths = inference_data_paths

    def import_tweets(self, path, columns, delete_columns, delimiter=','):
        """A function that takes a file path and converts it into a pandas frame
        containing only the necessary information for training an ML algorithm.

        Args:
            path (str): The path of the file that has to be converted into a
                pandas frame.
            columns (list): The columns that will be prepended to the data
                frame.
            delete_columns (list): The columns that will be deleted from the
                data frame.
            delimiter (str): The delimiter of the data present in the file.

        Returns:
            dataset (pandas.frame): A pandas that is produced from a file path
        """
        dataset = pd.read_csv(path, encoding='utf-8', delimiter=delimiter,
                              error_bad_lines=False, lineterminator='\n')
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
        """A function that by using the paths provided by the class imports all
        the datasets as pandas frames. It also removes the unnecessary columns
        from the training and inference dataset and returns these pandas
        frames.

        Returns:
            train_dataset (pandas.frame): The dataset that will be used to
                train the classifier.
            inference_dataset (list): A list of pandas frames that will be used
                for inference.
        """
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
        """A function that takes a tweet and applies a series of tranformation
        so as to have the text as clean as possible for better vectorization
        afterwards.

        Args:
            tweet (string): The text body of a tweet

        Returns:
            preprocessed_tweet (string): The text body of a preprocessed tweet
        """
        tweet.lower()
        # convert all urls to string "URL"
        tweet = re.sub(r'((www\.[^\s]+)|(https?://[^\s]+))', 'URL', tweet)
        # convert all @username to "AT_USER"
        tweet = re.sub(r'@[^\s]+', 'AT_USER', tweet)
        # correct all multiple white spaces to a single white space
        tweet = re.sub(r'[\s]+', ' ', tweet)
        # remove punctuation
        tweet = re.sub(r'[^\w\s]', '', tweet)
        # convert "#topic" to just "topic"
        tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
        # stemming and lemmatizing
        stemmer = PorterStemmer()
        lemmatizer = WordNetLemmatizer()
        temp_tweet = word_tokenize(tweet)
        tweet = []
        for word in tweet:
            temp_tweet.append(lemmatizer.lemmatize(stemmer.stem(word)))

        preprocessed_tweet = ' '.join(temp_tweet)
        return preprocessed_tweet

    def convert_to_format_extract_features(self, train_dataset,
                                           inference_datasets,
                                           method, args):
        """A function that takes a train set as a pandas frame and as a first
        step extracts the text and the sentiment. Afterwards fits a vectorizer
        to the train set and transforms and returns the whole training and
        inference set using that vectorizer.

        Args:
            train_dataset (pandas.frame): The tweets in the training dataset.
            inference_datasets (list): A list of pandas frames that contain the
                inference datasets from the various brexit dates.
            method (class): The method that is going to be used for
                vectorization.
            args (dict): The arguments that are going to be used by the
                vectorizer.

        Returns:
            train_features (np.array): The array of vectors that constitutes
                the training and test set.
            labels (np.array): The labels of the train and test set.
            inference_features (list): A list of np.arrays that has the
                inference set.
        """
        train_data = np.array(train_dataset.text)
        labels = np.array(train_dataset.sentiment)
        # take advantage of the fact that the same random state produces the
        # same split and shuffle procedure in scikit-learn
        X_train, X_test, y_train, y_test = train_test_split(
                train_data, labels, test_size=0.2, random_state=42)
        vector_transformer = method(**args)
        vector_transformer.fit(X_train)
        train_features = vector_transformer.transform(train_data)
        inference_features = []
        for inference_dataset in inference_datasets:
            inference_features.append(
                    vector_transformer.transform(inference_dataset.text))
        return train_features, labels, inference_features

    def train_and_evaluate_classifier(self, train_features, labels,
                                      classifier, classifier_args,
                                      scaler):
        """A function that takes the features, splits them into a train and
        test set and does the same split for the labels. Afterwards, it trains
        a model on the train set and a scaler if it's provided and returns some
        metrics about the model performance, as well as the trained model and
        scaler.

        Args:
            train_features (np.array): An array that contains the features that
                the classifier needs to be trained on.
            labels (np.array): An array that contains the labels that the
                classifier needs to be trained on.
            classifier (class): A class indicating the classifier that needs to
                be trained.
            classifier_args (dict): Arguments that need to be used by the
                 classifier class.
            scaler (class): A scaler that performs feature scaling.

        Returns:
            accuracy (float): The accuracy of the trained classifier on the
                test set.
            f1 (float): The F1 score of the trained classifier on the test set.
            model (class): The trained model.
            scaler (class): The trained scaler.
        """
        X_train, X_test, y_train, y_test = train_test_split(
                train_features, labels, test_size=0.2, random_state=42)
        model = classifier(**classifier_args)
        if scaler:
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='macro')
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy, f1, model, scaler

    def evaluate_on_brexit(self, model, inference_features, scaler):
        """A function that takes a trained model, a trained scaler and a list of
        features and applies the trained model (and scaler) to the features in
        order to perform inference.

        Args:
            model (class): A trained ML model.
            inference_features (list):  A list that contains arrays that
                represent the features calculated from the various Brexit
                dates.
            scaler (class): A trained scaler for the features.
        """
        for i, inference_feature in enumerate(inference_features):
            dataset = pd.read_csv(inference_dataset_paths[i],
                                  encoding='utf-8',
                                  error_bad_lines=False,
                                  lineterminator='\n')
            dataset.columns = self.inference_dataset_columns
            if scaler:
                inference_feature = scaler.tranform(inference_feature)
            y_pred = model.predict(inference_feature)
            for i in range(len(list(dataset.text))):
                print("{ \"", list(dataset.text)[i], "\":", y_pred[i], "},")


if __name__ == '__main__':
    # Take all the necessary paths
    inference_dataset_paths = [
        "../../data/test22-02-2016.csv",
        "../../data/test23-06-2016.csv",
        "../../data/test19-06-2017.csv",
        "../../data/test14-11-2018.csv",
        "../../data/test25-11-2018.csv",
        "../../data/test15-01-2019.csv"]
    # Initialize an instance of the class with the paths
    twitterSentiment = MLTweetSentiment(
            "../../data/twitter_data.csv",
            inference_dataset_paths
            )

    # Preprocess the tweets
    train_dataset, inference_datasets = twitterSentiment.preprocess_tweets()
    # Set up your grid
    count_args = {'stop_words': 'english'}
    features_extractors = [CountVectorizer]
    features_extractors_args = [count_args]
    classifiers = [XGBClassifier]
    xgb_args = {'max_depth': [13], 'n_estimators': [2000],
                'n_jobs': [16], 'learning_rate': [0.2]}
    classifier_args = [xgb_args]
    # Run the grid and print the score as well as the brexit tweets with
    # annotation
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
                scaler = None
                if isinstance(classifier, SVC):
                    scaler = StandardScaler(with_mean=False)
                accuracy, f1, model, scaler = \
                    twitterSentiment.train_and_evaluate_classifier(
                        train_features,
                        labels,
                        classifier,
                        possibility,
                        scaler
                        )
                print("Feature Extractor: {0}\tModel: {1}\t"
                      "Parameters: {2}\tF1 Score: {3}\tAccuracy:"
                      "{4}".format(feature_extractor, classifier,
                                   possibility, f1, accuracy))
                print("============================================================================================================")
                twitterSentiment.evaluate_on_brexit(model, inference_features,
                                                    scaler)
