from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import random
import csv

path = "/Users/dimkou/Documents/ETH/4th_semester/gess/" + \
       "brexit-twitter-sentiment-analysis/data/news_tweets_50k.csv"
data = []
data_labels = []
with open(path) as f:
    csv_reader = csv.reader(f, delimiter=',')
    for row in csv_reader:
        if row[2] != '0':
            data.append(row[1])
            data_labels.append(row[2])

print("Read all the data")
vectorizer = CountVectorizer(
    analyzer='word',
    lowercase=False,
)

features = vectorizer.fit_transform(
    data
)

features_nd = features.toarray()
print("Computed the features")

X_train, X_test, y_train, y_test = train_test_split(
        features_nd,
        data_labels,
        train_size=0.80,
        random_state=1234)

log_model = LogisticRegression()
log_model = log_model.fit(X=X_train, y=y_train)
print("Trained the classifier")
y_pred = log_model.predict(X_test)
print("Predicted in the test set")

j = random.randint(0, len(X_test)-7)
for i in range(j, j+7):
    print(y_pred[0])
    ind = features_nd.tolist().index(X_test[i].tolist())
    print(data[ind].strip())

print(accuracy_score(y_test, y_pred))
