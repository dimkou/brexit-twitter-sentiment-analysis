import pickle
import datetime
import pandas as pd


class DatasetLoader:
    def __init__(self, path):
        # Accepts both .csv and .p (pickle) file.
        # It will automatically convert to csv if pickle file is given
        self.path = path

    def loadData(self):
        pickle_data = self._checkAndLoadPickle()
        if pickle_data:
            self._convertToCsv(pickle_data)

        df = pd.read_csv(self.path)
        return list(zip(df["Date"], df["Text"], df["Score"]))

    def _convertToCsv(self, binary_file):
        res = {"Date": [], "Text": [], "Score": []}

        for k, v in binary_file.items():
            tweets = v["tweets"]
            for t in tweets:
                date = datetime.datetime.fromtimestamp(
                    t["timestamp_ms"] / 1000.0)
                date = date.strftime('%d-%m-%Y')
                res["Date"].append(str(date))
                res["Text"].append(t["text"])
                res["Score"].append(t["sentiment_score"])

        df = pd.DataFrame(res, columns=['Date', 'Text', 'Score'])
        csv = "{}.csv".format(self.path.rsplit('.', 1)[0])
        df.to_csv(csv, index=False)
        self.path = csv

    def _checkAndLoadPickle(self):
        try:
            with open(self.path, "rb") as f:
                file = pickle.load(f)
            return file
        except pickle.UnpicklingError:
            return None


if __name__ == "__main__":
    dl = DatasetLoader('/home/ageorgiou/Downloads/news_tweets_50k.p')
    print(dl.loadData())
