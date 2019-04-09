import tweepy
import json
from datetime import datetime

def lookup_tweets(tweet_IDs, api):
	full_tweets = []
	tweet_count = len(tweet_IDs)
	start_time = datetime.now()
	try:
		for i in range(int(tweet_count / 100) + 1):
			# Catch the last group if it is less than 100 tweets
			end_loc = min((i + 1) * 100, tweet_count)
			try:
				full_tweets.extend(
					api.statuses_lookup(id_=tweet_IDs[i * 100:end_loc])
				)
			except:
				print("Error occured at i={}".format(i))
				pass
			if i!=0 and i%500 == 0:
				print("{}/{} tweets retrieved, time elapsed: {}".format(i*100, tweet_count, datetime.now() -start_time))
		return full_tweets
	except :
		print("{}/{} tweets retrieved".format(i*100, tweet_count))
		print('Something went wrong, quitting...')

consumer_key = 'xxx'
consumer_secret = 'xxx'
access_token = 'xxx'
access_token_secret = 'xxx'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

file = open("./dataset/nepal/2015_Nepal_Earthquake_unlabelled_ids.txt")

print("Reading tweet ids")
TweetIDs = file.read().split("\n")

print("Looking for tweets")
results = lookup_tweets(TweetIDs, api)
print("Tweets retrieved: {}".format(len(results)))

tweet_dict = dict()

for tweet in results:
	if tweet:
		tweet_dict[tweet.id_str] = tweet.text

print("Saving dictionary: {}".format(len(tweet_dict.keys())))
with open("./dataset/nepal/2015_Nepal_Earthquake_unlabelled.txt", "w") as f:
	json.dump(tweet_dict, f)