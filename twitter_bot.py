#coding: utf-8

import os 
import tweepy
import time
import predict
import sqlite3
import pickle
import random
import tensorflow as tf 
import twitter_listener

book_list = [' https://www.amazon.co.jp/dp/4862760856/ref=cm_sw_r_tw_dp_x_779IzbMGRYVJS ',
            ' https://www.amazon.co.jp/dp/B01GJOQSO2/ref=cm_sw_r_tw_dp_x_I99Izb2YHWR9V ',
            ' https://www.amazon.co.jp/dp/477597114X/ref=cm_sw_r_tw_dp_x_e99IzbVBPJYHX ',
            ' https://www.amazon.co.jp/dp/4061592998/ref=cm_sw_r_tw_dp_x_Y.9Izb1ZYZRFW ',
            ' https://www.amazon.co.jp/dp/B01AXRCDZ4/ref=cm_sw_r_tw_dp_x_T-9Izb2TMXPTG ',
            ' https://www.amazon.co.jp/dp/B01CZK0B2Y/ref=cm_sw_r_tw_dp_x_8a-IzbFVB3F43 ',
            ' https://www.amazon.co.jp/dp/4560093024/ref=cm_sw_r_tw_dp_x_fc-IzbGXF6FA1 ',
            ' https://www.amazon.co.jp/dp/4532190452/ref=cm_sw_r_tw_dp_x_Wd-Izb5HTQAAX ',
            ' https://www.amazon.co.jp/dp/4003420950/ref=cm_sw_r_tw_dp_x_9g-Izb5J15S60 ',
            ]

anger_list = [' よくも言ったなあああ!!💢💢💢 ピヨヨヨヨヨヨヨヨヨヨ💢💢💢💢',
            ' 頭にバナナぶっ刺すよ？？💢💢💢',
            ' たこの入っていないたこ焼きみたいなあんたに言われたくないよ'
            ]
serif_list = [' ありがとう。',
            ' お疲れ様。今日も一日，よく頑張ったね。',
            ' ゆっくり休んで。',
            ' そのツイートって新規性あります？',
            ' 元気そうでよかった。']
def select_next_tweets():
    conn = sqlite3.connect('tweets.db')
    c = conn.cursor()
    c.execute("select sid, data, bot_flag from tweets where processed = 0")
    for row in c:
        sid  = row[0]
        data = pickle.loads(row[1])
        bot_flag = row[2]
        return sid, data, bot_flag
    return None, None, None

def mark_tweet_processed(status_id):
    conn = sqlite3.connect(twitter_listener.DB_NAME)
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

def is_contain(twit, str):
    if twit.find(str) != -1:
        return True
    return False

def post_reply(api, bot_flag, reply_body, screen_name, status_id):
    reply_body = reply_body.replace("_UNK", '💩')
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
    
def special_reply(api, bot_flag, screen_name, status_id, code):
    if code == 1:
        reply_text = random.choice(book_list) + "はおすすめ。"
    elif code == 2:
        reply_text = random.choice(anger_list)
    elif code == 3:
        reply_text = random.choice(serif_list)
        
    if bot_flag == twitter_listener.SHOULD_TWEET:
        print("My tweet:{0}".format(reply_text))
        if not reply_text:
            reply_text = '適切なお返事が応答できませんでした😇😇'
        api.update_status(status = reply_text)
    else:
        #reply_body = '適切なお返事が応答できませんでした😇😇'
        reply_text = "@" + screen_name + " " + reply_text
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

    with tf.Session(config = tf_config) as sess:
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
                    if is_contain(status.text, 'おすすめの本'):
                        special_reply(api, bot_flag, screen_name, status_id, code = 1)
                    elif is_contain(status.text, '人工無能'):
                        special_reply(api, bot_flag, screen_name, status_id, code = 2)
                    elif is_contain(status.text, 'ありがとう'):
                        special_reply(api, bot_flag, screen_name, status_id, code = 3)
                    else:
                        post_reply(api, bot_flag, reply_body, screen_name, status_id)
                except tweepy.TweepError as e:
                    if e.api_code == 187:
                        pass
                    else:
                        raise
            mark_tweet_processed(status_id)


if __name__ == "__main__":
    twitter_bot()