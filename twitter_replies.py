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

#認証情報を設定する
auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
api=tweepy.API(auth)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source_file', type = argparse.FileType('a'))
    parser.add_argument('target_file', type = argparse.FileType('a'))
    parser.add_argument('--languages', nargs = '+', default = ['ja'])
    args = parser.parse_args()

    while True:
        
    

if __name__ == "__main__":
    main()