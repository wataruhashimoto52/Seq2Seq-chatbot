import random 
import numpy as np 
import tensorflow as tf
from data_utils import *
from my_seq2seq import *
from six.moves import xrange
from six.moves import zip

class Seq2SeqModel(object):
    """
    Sequence-to-Sequence model with attention and for multiple buckets.abs

    This class implements a multi-layer recurrent neural network as encoder, 
    and an attention-based decoder.This is the same as the model described in
    this paper: https://arxiv.org/abs/1412.7499 .
    This class also allows to use GRU cells in addition to LSTM cells, and
    sampled softmax to handle large output vocabulary size. A single layer
    version of this model, but with bi-directional encoder, was presented in 
        https://arxiv.org/abs/1409.0473
    and sampled softmax is described in Section 3 of the following paper, 
        https://arxiv.org/abs/1412.2007
    """

    def __init__(self, source_vocab_size, target_vocab_size, buckets, size,
                    num_layers, max_gradient_norm, batch_size, learning_rate,
                    learning_rate_decay_factor, use_lstm = False,
                    num_samples = 1024, forward_only = False, beam_search = True, beam_size = 10, attention = True):
        """
        Create the model.


        Args:
            source_vocab_size: size of the source size.
            target_vocab_size: size of the target size.
            buckets: a list of pairs (I, 0), where I specifies maximum input length
                that will be processed in that bucket, and 0 specifies maximum output
                length. Training instances that have inputs longer than I or outputs
                longer than 0 will be pushed to the next bucket and padded accordingly.
                We assume that the list is sorted, e.g. [(2, 4), (8, 16)]
            size: number of units in each layer of the model.
            num_layers: number of layers in the model. 
            max_gradient_norm : gradients will be clipped to maximally this norm. 
            batch_size: the size of the batches used during training;
                the model construction is independent of batch_size, so it can be
                changed after initialization if this is convenient, e.g., for decoding.
            learning_rate: learning rate to start with.
            learning_rate_decay_factor: decay learning rate by this much when needed.
            use_lstm: if true, we use LSTM cells instead of GRU cells.
            num_samples: number of samples for sampled softmax.
            forward_only: if set, we do not construct the backward pass in the model.

        """
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.buckets = buckets
        self.batch_size = batch_size
        self.learning_rate = tf.Variable(float(learning_rate), trainable = False)
        self.learning_rate_decay_op = self.learning_rate.assign(
            self.learning_rate * learning_rate_decay_factor
        )
        self.global_step = tf.Variable(0, trainable = False)

        #If we use sampled softmax, we need an output projection.
        output_projection = None
        softmax_loss_function = None

        #Sampled softmax only makes sense if we sample less than vocabulary size.
        if num_samples > 0 and num_samples < self.target_vocab_size:
            with tf.device("/cpu:0"):
                w = tf.get_variable('proj_w', [size, self.target_vocab_size])
                w_t = tf.transpose(w)
                b = tf.get_variable('proj_b', [self.target_vocab_size])
            output_projection = (w, b)

            def sampled_loss(inputs, labels):
                with tf.device("/cpu:0"):
                    labels = tf.reshape(labels, [-1, 1])
                    #サンプリングされたsoftmax trainingの損失を計算して返す
                    return tf.nn.sampled_softmax_loss(w_t, b, labels, inputs, num_samples,
                                                        self.target_vocab_size)
            
            softmax_loss_function = sampled_loss
        #Create the internal multi-layer cell for out RNN.
        print('##### tf.get_variable_scope().reuse : {}'.format(tf.get_variable_scope().reuse))
        def gru_cell():
            return tf.contrib.rnn.GRUCell(size, reuse = tf.get_variable_scope().reuse)
        def lstm_cell():
            return tf.contrib.rnn.BasicLSTMCell(size, reuse = tf.get_variable_scope().reuse)

        single_cell = lstm_cell
        if use_lstm:
            single_cell = lstm_cell
        cell = single_cell()

        if num_layers > 1:
            cell_1 = tf.contrib.rnn.MultiRNNCell([single_cell() for _ in range(num_layers)], state_is_tuple = True)
            cell_2 = tf.contrib.rnn.MultiRNNCell([single_cell() for _ in range(num_layers)], state_is_tuple = True)

        #The seq2seq function: we use embedding for the input and attention.
        print("##### num_layers:{} #####".format(num_layers))
        print("##### {} #####".format(output_projection))
        def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
            if attention:
                print("Attention Model")
                return embedding_attention_seq2seq(encoder_inputs, decoder_inputs, cell,
                                            num_encoder_symbols = source_vocab_size,
                                            num_decoder_symbols = target_vocab_size,
                                            embedding_size = size,
                                            output_projection=output_projection,
                                            feed_previous=do_decode,
                                            beam_search=beam_search,
                                            beam_size = beam_size)
            
            else:
                print("Simple Model")
                return embedding_rnn_seq2seq(encoder_inputs, decoder_inputs, cell,
                                            num_encoder_symbols = source_vocab_size,
                                            num_decoder_symbols = target_vocab_size,
                                            embedding_size = size,
                                            output_projection=output_projection,
                                            feed_previous=do_decode,
                                            beam_search=beam_search,
                                            beam_size = beam_size)
        #Feed for inputs.
        
