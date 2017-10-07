#coding: utf-8

import sys 
import tensorflow as tf 
import numpy as np 
import train
import data_processor

def get_prediction(session, model, enc_vocab, rev_dec_vocab, text):
    token_ids = data_processor.sentence_to_token_ids(text, enc_vocab)
    bucket_id = min([b for b in range(len(data_processor.buckets))
                        if data_processor.buckets[b][0] > len(token_ids)])
    encoder_inputs, decoder_inputs, target_weights = model.get_batch({bucket_id:[(token_ids, [])]}, bucket_id)

    _, _, output_logits = model.step(session, encoder_inputs, decoder_inputs,
                                                target_weights, bucket_id, True, beam_search = False)
    
    outputs = [int(np.argmax(logit, axis = 1)) for logit in output_logits]
    if data_processor.EOS_ID in outputs:
        outputs = outputs[:outputs.index(data_processor.EOS_ID)]
    text = "".join([tf.compat.as_str(rev_dec_vocab[output]) for output in outputs])
    return text
    

def get_beam_search_prediction(session, model, enc_vocab, rev_dec_vocab, text):
    max_len = data_processor.buckets[-1][0]
    target_text = text
    if len(text) > max_len:
        target_text = text[:max_len]
    token_ids = data_processor.sentence_to_token_ids(target_text, enc_vocab)
    target_buckets = [b for b in range(len(data_processor.buckets))
                        if data_processor.buckets[b][0] > len(token_ids)]
    if not target_buckets:
        return []

    bucket_id = min(target_buckets)
    encoder_inputs, decoder_inputs, target_weights = model.get_batch({bucket_id:[(token_ids, [])]}, bucket_id)

    path, symbol, output_logits = model.step(session, encoder_inputs, decoder_inputs,
                                                target_weights, bucket_id, True, beam_search = data_processor.beam_search)
    beam_size = data_processor.beam_size
    k = output_logits[0]
    paths = []
    for kk in range(beam_size):
        paths.append([])
    curr = list(range(beam_size))
    num_steps = len(path)
    for i in range(num_steps - 1, -1, -1):
        for kk in range(beam_size):
            paths[kk].append(symbol[i][curr[kk]])
            curr[kk] = path[i][curr[kk]]
    recos = set()
    ret = []
    i = 0
    for kk in range(beam_size):
        foutputs = [int(logit) for logit in paths[kk][::-1]]

        # If there is an EOS symbol in outputs, cut them at that point.
        if data_processor.EOS_ID in foutputs:
            #         # print outputs
            foutputs = foutputs[:foutputs.index(data_processor.EOS_ID)]
        rec = " ".join([tf.compat.as_str(rev_dec_vocab[output]) for output in foutputs])
        if rec not in recos:
            recos.add(rec)
#            print("reply {}".format(i))
#            i = i + 1
            ret.append(rec)
    return ret


class EasyPredictor:
    def __init__(self, session):
        self.session = session
        train.show_progress("Creating Model...")
        self.model = train.create_or_restore_model(self.session, data_processor.buckets, forward_only = True,
                                                     beam_search = data_processor.beam_search, beam_size = data_processor.beam_size)
        self.model.batch_size = 1
        train.show_progress("done\n")
        self.enc_vocab, _ = data_processor.initialize_vocabulary(data_processor.VOCAB_ENC_PATH)
        _, self.rev_dec_vocab = data_processor.initialize_vocabulary(data_processor.VOCAB_DEC_PATH)

    def predict(self, text):
        if data_processor.beam_search:
            replies = get_beam_search_prediction(self.session, self.model, self.enc_vocab, self.rev_dec_vocab, text)
            return replies
        else:
            reply = get_prediction(self.session, self.model, self.enc_vocab, self.rev_dec_vocab, text)
            return [reply]

def predict():
    tf_config = tf.ConfigProto(gpu_options = tf.GPUOptions(visible_device_list = "0"))
    with tf.Session(config=tf_config) as sess:

        predictor = EasyPredictor(sess)
        sys.stdout.write("> ")
        sys.stdout.flush()
        line = sys.stdin.readline()
        while line:
            replies = predictor.predict(line)
            #for i, text in enumerate(replies):
                #print(i, text)
            print(replies[0])
            print("> ", end = "")
            sys.stdout.flush()
            line = sys.stdin.readline()

if __name__ == "__main__":
    predict()
