from textblob import TextBlob
import tweepy
import pandas as pd

# Authenticate Twitter API credentials
consumer_key = 'Yl8CjjV6flI4s87TqbXLnLT3f'
consumer_secret = 'ZoizcO380hqGTZ3FVIDhkkCfr6zDoIlt0cjt6WGCgxcRPssy2p'
access_token = '1426913728583258115-rSIKTw8RUAGziu2P87vNyJU2rUrfx1'
access_token_secret = 'SoVSxHsq88hSmWAq8K0IfMZg4ZQU74SvXv2VN2Ewc67eL'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

# Create API object
api = tweepy.API(auth)

def extract_negative_reason(tweet_text):
    """
    Returns the negative reason in the tweet text.
    """
    negative_reasons = ['delay', 'cancellation', 'lost luggage', 'bad service']
    for reason in negative_reasons:
        if reason in tweet_text.lower():
            return reason
    return None

keyword = '@united'
geocode = '40.7128,-74.0060,200km'
tweet_count = 1000

tweets = tweepy.Cursor(api.search_tweets, q=keyword, lang='en', geocode=geocode, tweet_mode='extended').items(tweet_count)

data = []
latitude_list = []
longitude_list = []

for tweet in tweets:
    tweet_id = tweet.id_str
    tb = TextBlob(tweet.full_text)
    airline_sentiment = tb.sentiment.polarity
    airline_sentiment_confidence = tb.sentiment.subjectivity
    negative_reason = extract_negative_reason(tweet.full_text)
    negative_reason_confidence = tb.sentiment.subjectivity
    airline_name = tweet.author.name
    user_name = tweet.author.screen_name
    retweet_count = tweet.retweet_count
    tweet_text = tweet.full_text
    tweet_created_time = tweet.created_at

    if tweet.coordinates:
        latitude = tweet.coordinates['coordinates'][1]
        longitude = tweet.coordinates['coordinates'][0]
    elif tweet.place:
        latitude = tweet.place.bounding_box.coordinates[0][0][1]
        longitude = tweet.place.bounding_box.coordinates[0][0][0]
    else:
        latitude = None
        longitude = None
    

    data.append([tweet_id, airline_sentiment, airline_sentiment_confidence, negative_reason, negative_reason_confidence,
                 airline_name, user_name, retweet_count, tweet_text, tweet_created_time, latitude, longitude])

columns = ['Tweet ID', 'Airline Sentiment', 'Airline Sentiment Confidence', 'Negative Reason', 'Negative Reason Confidence',
           'Airline Name', 'User Name', 'Retweet Count', 'Tweet Text', 'Tweet Created Time', 'Latitude', 'Longitude']
df = pd.DataFrame(data, columns=columns)
print(df)
#df.to_csv('extracted_tweets1.csv', index=False)