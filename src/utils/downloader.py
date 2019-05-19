from twitterscraper import query_tweets
import twitterscraper.query
import datetime as dt
import pandas as pd
import random
import string


def randomString(stringLength=10):
    """Generate a random string of fixed length """
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))


class Twitter:
    def __init__(self):
        pass

    def getTweets(self, query, begin_date, end_date, limit=None, n_jobs=2):
        """A function that takes a query and some dates and scrapes all tweets
        from Twitter that satisfy the given criteria.

        Args:
            query (str): The search string.
            begin_date (string): The start date that tweets are going to be
                scraped.
            end_date (string): The start date that tweets are going to be
                scraped.
            limit (string): A threshold number of tweets downloaded
            n_jobs (int): The degree of parallelism.

        Returns:
            text_list (list): A list of the scraped tweets.
        """
        # Date format is DD-MM-YYY (string)
        twitterscraper.query.HEADER = {
            'User-Agent': random.choice(twitterscraper.query.HEADERS_LIST)
        }
        b_day, b_month, b_year = begin_date.split('-')
        e_day, e_month, e_year = end_date.split('-')
        tweets = query_tweets(query,
                              begindate=dt.date(int(b_year), int(b_month),
                                                int(b_day)),
                              enddate=dt.date(int(e_year), int(e_month),
                                              int(e_day)),
                              limit=limit,
                              lang='en',
                              poolsize=n_jobs)
        text_list = [(str(t.timestamp.strftime('%d-%m-%Y')),
                      t.text.strip().replace('\n', ' ')) for t in tweets]
        return text_list

    def getTweetsFromDates(self,
                           query,
                           list_of_dates,
                           limit_per_day=2000,
                           n_jobs=2):
        """A function that takes a query and some dates and scrapes all tweets
        from Twitter that satisfy the given criteria.

        Args:
            query (str): The search string.
            list_of_dates (list): The dates that we are going to scrape
                Twitter.
            limit_per_day (int): A threshold number of tweets downloaded per
                date.
            n_jobs (int): The degree of parallelism.

        Returns:
            res (list): A list of the scraped tweets.
        """
        # Date format is DD-MM-YYYY (string)
        res = []
        for date in list_of_dates:
            b_day, b_month, b_year = date.split('-')
            begin_date = dt.date(int(b_year), int(b_month), int(b_day))
            end_date = begin_date + dt.timedelta(days=1)
            end_date = str(end_date.strftime('%d-%m-%Y'))
            res += self.getTweets(query, date, end_date, limit_per_day, n_jobs)
        return res

    def writeCSV(self, tweets, out_path):
        """A function that takes a list of tweets and writes them to a csv.

        Args:
            tweets (list): A list of tweets with their dates.
            out_path (string): The path that the tweets are going to be
            written.
        """
        res = {"Date": [], "Text": [], "Score": []}
        for t in tweets:
            res["Date"].append(t[0])
            res["Text"].append(t[1])
            res["Score"].append(0.0)
        df = pd.DataFrame(res, columns=['Date', 'Text', 'Score'])
        df.to_csv(out_path, index=False)


if __name__ == '__main__':
    for i in range(3000):
        twitterscraper.query.HEADERS_LIST.append(
            f'MyBrowser{randomString(10)}/9.80 (X11)')
    twitter = Twitter()
    # dates = [
    #     "22-02-2016", "23-06-2016", "19-06-2017", "14-11-2018", "25-11-2018",
    #     "15-01-2019"
    # ]
    dates = ['23-06-2016']
    for date in dates:
        tweets = []
        while len(tweets) == 0:
            tweets = twitter.getTweetsFromDates('brexit', [date],
                                                limit_per_day=40000,
                                                n_jobs=4)
        twitter.writeCSV(tweets, f"test{date}.csv")
