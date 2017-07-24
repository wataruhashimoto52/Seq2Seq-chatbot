#coding: utf-8

import tweepy
import os
import re
import argparse
import sys
import time
import traceback


CONSUMER_KEY = os.environ['CONSUMER_KEY']
CONSUMER_SECRET = os.environ['CONSUMER_SECRET']
ACCESS_TOKEN = os.environ['ACCESS_TOKEN']
ACCESS_TOKEN_SECRET = os.environ['ACCESS_TOKEN_SECRET']

class ReplyStreamListener(tweepy.StreamListener):
    def __init__(self, api, target_file, source_file):
        super().__init__(api)
        self.target_file = target_file
        self.source_file = source_file
        self.statuses = []
        self.username = re.compile('@\w{,15}')
        self.hashtag = re.compile('#\w+')
        self.retweet = re.compile('RT.?:?')
        self.url = re.compile('http\S+')

    def clean_twitter(self, text):
        text = self.username.sub('', text)
        text = self.hashtag.sub('', text)
        text = self.retweet.sub('', text)
        text = self.url.sub('', text)
        text = text.replace('\n', ' ')
        text = text.replace('\r', ' ')
        text = text.replace('\t', ' ')
        text = text.strip()
        return text

    def on_status(self, status):
        
        #返信先ツイートIDがあるなら，statusにappend
        if status.in_reply_to_status_id_str is not None:
            self.statuses.append(status)

            #accumulate 100 statuses and lookup batch of in_reply_to
            if len(self.statuses) == 100:
                in_reply_to_ids = [status.in_reply_to_status_id_str for status in self.statuses]
                
                #Note: response is smaller than request if some tweets were private
                source_statuses = self.api.statuses_lookup(in_reply_to_ids, trim_user = True)

                #construct dictionary of source id to target text
                target_dictionary = {status.in_reply_to_status_id_str : status.text for status in self.statuses}

                for source_status in source_statuses:
                    #lookup target text from source id
                    target_text = target_dictionary[source_status.id_str]

                    source_text = self.clean_twitter(source_status.text)
                    target_text = self.clean_twitter(target_text)

                    #save to text file
                    print(source_text, file = self.source_file)
                    print(target_text, file = self.target_file)
                
                self.statuses = []
                self.source_file.flush()
                self.target_file.flush()
                print('Collected', len(source_statuses), 'pairs')
    
    def on_error(self, status_code):
        print('Stream error with status code:', status_code, file = sys.stderr)
        return False



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source_file', type = argparse.FileType('a'))
    parser.add_argument('target_file', type = argparse.FileType('a'))
    parser.add_argument('--languages', nargs = '+', default = ['ja'])
    args = parser.parse_args()

    while True:
        try:
            auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
            auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
            api = tweepy.API(auth)
            reply_stream_listener = ReplyStreamListener(api, args.target_file, args.source_file)
            reply_stream = tweepy.Stream(auth = api.auth, listener = reply_stream_listener)
            reply_stream.sample(languages = args.languages)

        except:
            traceback.print_exc(limit = 10, file = sys.stderr, chain = False)
            time.sleep(10)
            continue
    

if __name__ == "__main__":
    main()