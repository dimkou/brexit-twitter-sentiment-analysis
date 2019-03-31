from twitterscraper import query_tweets
import datetime as dt
import pandas as pd


class Twitter:
    def __init__(self, n_jobs=2):
        self.n_jobs = n_jobs

    def getTweets(self, query, begindate, enddate, limit=None):
        # Date format is DD-MM-YYY (string)
        bday, bmonth, byear = begindate.split('-')
        eday, emonth, eyear = enddate.split('-')
        tweets = query_tweets(
            query,
            begindate=dt.date(int(byear), int(bmonth), int(bday)),
            enddate=dt.date(int(eyear), int(emonth), int(eday)),
            limit=limit,
            lang='en',
            poolsize=self.n_jobs)
        textlist = [(str(t.timestamp.strftime('%d-%m-%Y')), t.text.strip())
                    for t in tweets]
        return textlist

    def getTweetsOneDay(self, query, date, limit=None):
        # Date format is DD-MM-YYYY (string)
        bday, bmonth, byear = date.split('-')
        begindate = dt.date(int(byear), int(bmonth), int(bday))
        enddate = begindate + dt.timedelta(days=1)
        enddate = str(enddate.strftime('%d-%m-%Y'))
        return self.getTweets(query, date, enddate, limit)

    def writeCSV(self, query, list_of_dates, outpath, limit_per_day=2000):
        res = {"Date": [], "Text": [], "Score": []}
        for date in list_of_dates:
            tweetlist = self.getTweetsOneDay(query, date, limit=limit_per_day)
            for t in tweetlist:
                res["Date"].append(t[0])
                res["Text"].append(t[1])
                res["Score"].append(0.0)
        df = pd.DataFrame(res, columns=['Date', 'Text', 'Score'])
        df.to_csv(outpath, index=False)


if __name__ == '__main__':
    twitter = Twitter(n_jobs=20)
    dates = [
        "22-02-2016", "23-06-2016", "19-06-2017", "14-11-2018", "25-11-2018",
        "15-01-2019"
    ]
    twitter.writeCSV("brexit", dates, "test.csv", limit_per_day=None)
