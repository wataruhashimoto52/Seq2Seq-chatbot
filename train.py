import os
import sys
import math
import random 
import numpy as np 
import tensorflow as tf 
import data_processor
import seq2seq_model 

def show_progress(text):
    sys.stdout.write(text)
    sys.stdout.flush()

def read_data_into_buckets(enc_path, dec_path, buckets):
    """
    Read tweets and reply and put them into buckets based on their length.
    """

    data_set = [[] for _ in buckets]
    with tf.gfile.GFile(enc_path, mode='r') as ef, tf.gfile.GFile(dec_path, mode='r') as df:
        #read tweets and replies from text file.
        tweet, reply = ef.readline(), df.readline()
        counter = 0
        while tweet and reply:
            counter += 1
            if counter % 100000 == 0:
                print('   Reading data line %d' % counter)
                sys.stdout.flush()
            source_ids = [int(x) for x in tweet.split()]
            target_ids = [int(x) for x in reply.split()]
            target_ids.append(data_processor.EOS_ID)
            for bucket_id, (source_size, target_size) in enumerate(buckets):
                #Find bucket to put this conversation based on tweet and reply length
                if len(source_ids) < source_size and len(target_ids) < target_size:
                    data_set[bucket_id].append([source_ids, target_ids])
                    break
            tweet, reply = ef.readline(), df.readline()
    for bucket_id in range(len(buckets)):
        print("{}={}=".format(buckets[bucket_id], len(data_set[bucket_id])))
    return data_set

def create_or_restore_model(session, buckets, forward_only, beam_search, beam_size):
    
    # beam search is off for training
    """Create model and initialize or load parameters"""

    model = seq2seq_model.Seq2SeqModel(source_vocab_size=config.MAX_ENC_VOCABULARY,
                                       target_vocab_size=config.MAX_DEC_VOCABULARY,
                                       buckets=buckets,
                                       size=config.LAYER_SIZE,
                                       num_layers=config.NUM_LAYERS,
                                       max_gradient_norm=config.MAX_GRADIENT_NORM,
                                       batch_size=config.BATCH_SIZE,
                                       learning_rate=config.LEARNING_RATE,
                                       learning_rate_decay_factor=config.LEARNING_RATE_DECAY_FACTOR,
                                       beam_search=beam_search,
                                       attention=True,
                                       forward_only=forward_only,
                                       beam_size=beam_size)

    print("model initialized")
    ckpt = tf.train.get_checkpoint_state("./")
    # the checkpoint filename has changed in recent versions of tensorflow
    checkpoint_suffix = ".index"
    if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path + checkpoint_suffix):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
    return model


def next_random_bucket_id(buckets_scale):
    n = np.random.random_sample()
    bucket_id = min([i for i in range(len(buckets_scale)) if buckets_scale[i] > n])
    return bucket_id


def train():
    with tf.Session(config=tf_config) as sess:

        show_progress("Setting up data set for each buckets...")
        train_set = read_data_into_buckets(data_processor.TRAIN_ENC_IDX_PATH, data_processor.TRAIN_DEC_IDX_PATH, data_processor.buckets)
        valid_set = read_data_into_buckets(data_processor.VAL_ENC_IDX_PATH, data_processor.VAL_DEC_IDX_PATH, data_processor.buckets)
        show_progress("done\n")

        show_progress("Creating model...")
        # False for train
        beam_search = False
        model = create_or_restore_model(sess, data_processor.buckets, forward_only=False, beam_search=beam_search, beam_size=data_processor.beam_size)

        show_progress("done\n")

        # list of # of data in ith bucket
        train_bucket_sizes = [len(train_set[b]) for b in range(len(data_processor.buckets))]
        train_total_size = float(sum(train_bucket_sizes))

        # Originally from https://github.com/1228337123/tensorflow-seq2seq-chatbot
        # This is for choosing randomly bucket based on distribution
        train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                               for i in range(len(train_bucket_sizes))]

        show_progress("before train loop")
        # Train Loop
        steps = 0
        previous_perplexities = []
        writer = tf.summary.FileWriter(data_processor.LOGS_DIR, sess.graph)

        while True:
            bucket_id = next_random_bucket_id(train_buckets_scale)
#            print(bucket_id)

            # Get batch
            encoder_inputs, decoder_inputs, target_weights = model.get_batch(train_set, bucket_id)
            #      show_progress("Training bucket_id={0}...".format(bucket_id))

            # Train!
            _, average_perplexity, _ = model.step(sess, encoder_inputs, decoder_inputs, target_weights,
                                                           bucket_id,
                                                           forward_only=False,
                                                           beam_search=beam_search)
#            _, average_perplexity, ,summary, _ = model.step(sess, encoder_inputs, decoder_inputs, target_weights,
#                                                           bucket_id,
#                                                           forward_only=False,
#                                                           beam_search=beam_search)

            #      show_progress("done {0}\n".format(average_perplexity))

            steps = steps + 1
            if steps % 10 == 0:
#                writer.add_summary(summary, steps)
                show_progress(".")
            if steps % 500 != 0:
                continue

            # check point
            checkpoint_path = "seq2seq.ckpt"
            show_progress("Saving checkpoint...")
            model.saver.save(sess, checkpoint_path, global_step=model.global_step)
            show_progress("done\n")

            perplexity = math.exp(average_perplexity) if average_perplexity < 300 else float('inf')
            print ("global step %d learning rate %.4f perplexity "
                   "%.2f" % (model.global_step.eval(), model.learning_rate.eval(), perplexity))

            # Decrease learning rate if no improvement was seen over last 3 times.
            if len(previous_perplexities) > 2 and perplexity > max(previous_perplexities[-3:]):
                sess.run(model.learning_rate_decay_op)
            previous_perplexities.append(perplexity)

            for bucket_id in range(len(data_processor.buckets)):
                if len(valid_set[bucket_id]) == 0:
                    print("  eval: empty bucket %d" % bucket_id)
                    continue
                encoder_inputs, decoder_inputs, target_weights = model.get_batch(valid_set, bucket_id)
                _, average_perplexity, _ = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, True, beam_search=beam_search)
#                writer.add_summary(valid_summary, steps)
                eval_ppx = math.exp(average_perplexity) if average_perplexity < 300 else float('inf')
                print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))

if __name__ == "__train__":
    train()