Seq2Seq chatbot (ja)
====

Overview

## Description
This is seq2seq chatbot.

## Requirement
python 3.5.2  
Tensorflow r1.2  
Mecab  
tweepy  

## Demo

1. Prepare train data.

    1, Registration to the Twitter API(https://apps.twitter.com).

    2, Extraction of consumerkey, consumer secret key, access token key and access token secret key.

    3, Installation forego.

    `$ brew install forego`

    Then, please make `.env` file and write consumerkey, consumer secret key, access token key and access token secret key.

    `$ vi .env` 

    ```
    CONSUMER_KEY=...
    CONSUMER_SECRET=...
    ACCESS_TOKEN=...
    ACCESS_TOKEN_SECRET=...
    ```

    4, `$ forego run python twitter_replies.py`
2. Preprocess the train data and generate vocabulary files, ID files, and some ones.

    `$ python data_processor.py`  

3. Train seq2seq chatbot. When perplexity went down sufficiently and you think it's time to run, just ctrl-c to stop learning. 

    `$ python train.py`

4. Let's talk to him.  

    `$ python predict.py`

    
## Contribution

## Author

[wataruhashimoto52] https://github.com/wataruhashimoto52
