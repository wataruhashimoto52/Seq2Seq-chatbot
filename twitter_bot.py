#coding: utf-8

import os 
import tensorflow as tf 
import tweepy
import time
import predict
import sqlite3
import pickle
import twitter_listener

def select_next_tweets():
    conn = sqlite3.connect(tweet)
    c = conn.cursor()
    for row in c:
        sid  = row[0]
        data = pickle.loads(row[1])
        bot_flag = row[2]
        return sid, data, bot_flag
    return None, None, None

def mark_tweet_processed(status_id):
    conn = sqlite3.connect(tweet_listener.DB_NAME)
    c = conn.cursor()
    c.execute("update tweets set processed = 1 where sid = ?", [status_id])
    conn.commit()
    conn.close()

def tweets():
    while True:
        status_id, tweet, bot_flag = select_next_tweets()
        if status_id is not None:
            yield(status_id, tweet, bot_flag)
        time.sleep(1)

def post_reply(api, bot_flag, reply_body, screen_name, status_id):
    reply_body = reply_body.replace("_UNK", '◯')
    if bot_flag == twitter_listener.SHOULD_TWEET:
        reply_text = reply_body
        print("My tweet:{0}".format(reply_text))
        if not reply_text:
            reply_text = '適切なお返事が応答できませんでした😇😇'
        api.update_status(status = reply_text)
    else:
        if not reply_body:
            reply_body = '適切なお返事が応答できませんでした😇😇'
        reply_text = "@" + screen_name + " " + reply_body
        print("Reply:{0}".format(reply_text))
        api.update_status(status = reply_text, in_reply_to_status_id = status_id)
    

def twitter_bot():
    tf_config = tf.ConfigProto(gpu_options = tf.GPUOptions(visible_device_list = "0"))
    
    CONSUMER_KEY = os.environ['CONSUMER_KEY']
    CONSUMER_SECRET = os.environ['CONSUMER_SECRET']
    ACCESS_TOKEN = os.environ['ACCESS_TOKEN']
    ACCESS_TOKEN_SECRET = os.environ['ACCESS_TOKEN_SECRET']

    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
    api = tweepy.API(auth)

    with tf.Session(tf_config) as sess:
        predictor = predict.EasyPredictor(sess)

        for tweet in tweets():
            status_id, status, bot_flag = tweet
            print("Processing {0}...".format(status.text))
            screen_name = status.author.screen_name
            replies = predictor.predict(status.text)
            if not replies:
                print("no reply")
                continue

            reply_body = replies[0]
            if reply_body is None:
                print("No reply predicted")
            else:
                try:
                    post_reply(api, bot_flag, reply_body, screen_name, status_id)
                except tweepy.TweepError as e:
                    if e.api_code == 187:
                        pass
                    else:
                        raise
            mark_tweet_processed(status_id)


if __name__ == "__main__":
    twitter_bot()