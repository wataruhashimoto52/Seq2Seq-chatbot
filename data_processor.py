import re
import sys
import tensorflow as tf

#For Japanese tokenizer
import MeCab


#path list
SOURCE_PATH = "data/source.txt"
TARGET_PATH = "data/target.txt"

TRAIN_ENC_PATH = "generated/train_enc.txt"
VALIDATION_ENC_PATH = "generated/validation_enc.txt"

TRAIN_DEC_PATH = "generated/train_dec.txt"
VALIDATION_DEC_PATH = "generated/validation_dec.txt"

VOCAB_ENC_PATH = "generated/vocab_enc.txt"
VOCAB_DEC_PATH = "generated/vocab_dec.txt"

TRAIN_ENC_IDX_PATH = "generated/train_enc_idx.txt"
TRAIN_DEC_IDX_PATH = "generated/train_dec_idx.txt"
VAL_ENC_IDX_PATH = "generated/val_enc_idx.txt"
VAL_DEC_IDX_PATH = "generated/val_dec_idx.txt"

MAX_VOCABULARY = 50000
DIGIT_RE = re.compile(r"\d")
_WORD_SPLIT = re.compile("([.,!/?\":;)(])")

_PAD = "_PAD"
_GO = "_GO"
_EOS = "_EOS"
_UNK = "_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

tagger = MeCab.Tagger("-Owakati")

def japanese_tokenizer(sentence):
    #sentenceのtypeはstr??
    assert type(sentence) is str

    result = tagger.parse(sentence)
    return result.split()

def num_lines(file):
    """
    ファイル中の文章の行数を返す

    Args:
        file: target file
    
    Returns:
        of lines in file
    """

    return sum(1 for _ in open(file))

def create_train_validation(filepath, train_path, validation_path, train_ratio = 0.9):
    """
    ファイルパス先のファイルをtrainとvalidationデータに分ける

    Args:
      filepath: source file path
      train_path: path to write train data
      validation_path: path to write validation data
      train_ration = train data ratio

      returns None
    """

    nb_lines = num_lines(filepath)
    nb_train = int(nb_lines * train_ratio)
    counter = 0

    with tf.gfile.GFile(filepath, "r") as f, tf.gfile.GFile(train_path, "w") as trf, tf.gfile.GFile(validation_path, "w") as vlf:
        for line in f:
            if counter < nb_train:
                trf.write(line)
            else:
                vlf.write(line)

            counter = counter + 1

def create_vocabulary(filepath, vocabulary_path, max_vocabulary_size, tokenizer = japanese_tokenizer):
    """
    語彙ファイルを作る．
    """

    if tf.gfile.Exists(vocabulary_path):
        print("Found vocabulary file")
        return

    with tf.gfile.GFile(filepath, "r") as f:
        counter = 0
        vocab = {} #word:word_freq

        for line in f:
            counter += 1
            words = tokenizer(line)

            if counter % 5000 == 0:
                sys.stdout.write(".")
                sys.stdout.flush()
            for word in words:
                
                word = re.sub(DIGIT_RE, "0", word)

                if word in vocab:
                    vocab[word] += 1
                else:
                    vocab[word] = 1

        vocab_list = _START_VOCAB + sorted(vocab, key = vocab.get, reverse = True)
        if len(vocab_list) > max_vocabulary_size:
            #できるだけ打ち切りたくないからmax_vocab_sizeは大きめのほうが良いかも？？？
            vocab_list = vocab_list[:max_vocabulary_size]
        with tf.gfile.GFile(vocabulary_path, "w") as vocab_file:
            for w in vocab_list:
                vocab_file.write(w + "\n")
        print("\n")

def basic_tokenizer(sentence):
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(re.split(_WORD_SPLIT, space_separated_fragment))
    return [w for w in words in w]

def sentence_to_token_ids(sentence, vocabulary, tokenizer = japanese_tokenizer, normalize_digits = True):
    if tokenizer:
        words = tokenizer(sentence)
    else:
        words = basic_tokenizer(sentence)
    if not normalize_digits:
        return [vocabulary.get(w, UNK_ID) for w in words]
    return [vocabulary.get(w, UNK_ID) for w in words]

def data_to_token_ids(data_path, target_path, vocabulary_path,
             tokenizer = japanese_tokenizer, normalize_digits = True):
    if not tf.gfile.Exists(target_path):
        print("Tokenizing data in %s" % data_path)
        vocab, _ = initialize_vocabulary(vocabulary_path)
        with tf.gfile.GFile(data_path, "rb") as data_file:
            with tf.gfile.GFile(target_path, "wb") as tokens_file:
                counter = 0
                for line in data_file:
                    counter += 1
                    if counter % 1000000 == 0:
                        print("   tokenizing line %d" % counter)
                    
                    line = line.decode('utf-8')
                    token_ids = sentence_to_token_ids(line, vocab, tokenizer,
                                                        normalize_digits)
                    tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")
        

def initialize_vocabulary(vocabulary_path):
    if tf.gfile.Exists(vocabulary_path):
        rev_vocab = []
        with tf.gfile.GFile(vocabulary_path, "r") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip() for line in rev_vocab]

        #Dictionary of (word, idx)
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocabulary_path)


if __name__ == "__main__":
    print("Splitting into train and validation data...")
    #print(num_lines(SOURCE_PATH))
    #print(num_lines(TARGET_PATH))
    create_train_validation(SOURCE_PATH, TRAIN_ENC_PATH, VALIDATION_ENC_PATH)
    create_train_validation(TARGET_PATH, TRAIN_DEC_PATH, VALIDATION_DEC_PATH)
    print("Done")

    print("Creating vocabulary files...")
    create_vocabulary(SOURCE_PATH, VOCAB_ENC_PATH, MAX_VOCABULARY)
    create_vocabulary(TARGET_PATH, VOCAB_DEC_PATH, MAX_VOCABULARY)
    print("Done")

    print("Creating sentence idx files...")
    data_to_token_ids(TRAIN_ENC_PATH, TRAIN_ENC_IDX_PATH, VOCAB_ENC_PATH)
    data_to_token_ids(TRAIN_DEC_PATH, TRAIN_DEC_IDX_PATH, VOCAB_DEC_PATH)
    data_to_token_ids(VALIDATION_ENC_PATH, VAL_ENC_IDX_PATH, VOCAB_ENC_PATH)
    data_to_token_ids(VALIDATION_DEC_PATH, VAL_DEC_IDX_PATH, VOCAB_DEC_PATH)
    print("Done")