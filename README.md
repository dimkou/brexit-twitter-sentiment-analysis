# Sentiment Analysis of Tweets about Brexit
## Description
This is the repo for the project of ETH's GESS course [Data Science in Techno-Socio-Economic Systems](http://www.coss.ethz.ch/education/datascience.html). The goal of this projects is to apply various sentiment analysis ML techniques/algorithms to datasets found online and then collect tweets from dates that are important for the Brexit timeline and use them as a "test set" to do a quantification of people's tweets about Brexit.
###  Current important dates for Brexit
#### Events happened
1. February 22, 2016: Referendum announced
2. June 23, 2016: Referendum held
3. June 19, 2017: Negotiations between the UK and EU begin
4. November 14, 2018: Withdrawal agreement between the UK and EU published
5. November 25, 2018: EU endorses the aforementioned withdrawal agreement
6. January 15, 2019: UK parliament rejects the agreement

## Downloading the code
The code can be cloned from GitHub using:
```bash
git clone https://github.com/dimkou/brexit-twitter-sentiment-analysis.git && cd brexit-twitter-sentiment-analysis
```

## Downloading the training data
Due to the large size of the training data, they are not included in the repo. The training data can be downloaded from [here](https://polybox.ethz.ch/index.php/s/KAOUIyv3CfbgxHt) or by using the command:
```bash
wget https://polybox.ethz.ch/index.php/s/KAOUIyv3CfbgxHt/download -O data/twitter_data.csv.gz
```

Then we need to extract the data using the following command:
```bash
gzip -d data/twitter_data.csv.gz
```

## Installing prerequisites
In order to run our code, some libraries such as `scikit-learn` are required to be installed. All the prerequisites are included in the file `requirements-ml.txt` and can simply be installed using the command:
```bash
pip install -r requirements-ml.txt
```

## Running the code
Now that we have our training data and all prerequisites installed we are ready to run the code. Simply run the following command to train and evaluate all classifiers:
```bash
python src/ml/classify.py
```
*NOTE: Training all classifiers and searching for the optimal hyperparameters can take up to 3 days (on 16 cores).*

## Downloading Brexit tweets (optional)
All brexit tweets are already included in the repo. However, for completeness we explain how one can download them using `twitterscraper`. To this end, we have implemented `downloader.py` to simplify the process:
```bash
python src/utils/downloader.py && mv src/utils/test* data/
```
*NOTE: Downloading the tweets can take up to 2-3 hours.*

## Running the demo
All the prerequisites are included in the file `requirements-demo.txt` and can simply be installed using the command:
```bash
pip install -r requirements-demo.txt
```

Then simply go the folder where the file manage.py is located and run:
```bash
python manage.py runserver
```

Then open your browser at 127.0.0.1:8000 and experiment! On click in every date a tweet and its label will appear as a popup window. We have selected 100 random tweets from every date.