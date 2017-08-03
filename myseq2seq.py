#coding: utf-8

import os
import tensorflow as tf 
from six.moves import xrange 
from six.moves import zip

#tryなんとかを書く，予定．

def _extract_argmax_and_embed(embedding, output_projection = None,
                                update_embedding = True):
    """
    Get a loop_function that extracts the previous symbol and embeds it.
    """
    def loop_function(prev, _):
        if output_projection is not None:
            prev = 1

def rnn_decoder(decoder_inputs, initial_state, cell, loop_function = None,
                scope = None):
    
    """
    RNN decoder for the sequence-to-sequence model.

    Args:
    decoder_inputs: a list of 2D tensors [batch_size x input_size]
    initial_state: 2D tensor with shape [batch_size x cell.state_size]
    cell: rnn_cell.RNNCELL defining the cell function and size.
    loop_function: If not None, this function will be applied to the i-th output
        in order to generate the i+1 -st input, and decoder_inputs will be ignored,
        except for the first element ("Go" symbol).This can be used for decoding, 
        but also for training to emulate "Scheduled Sampling for Sequence Prediction
        with Recurrent Neural Networks".
    scope: Variablescope for the created subgraph; defaults to "rnn_decoder".

    returns:
    A tuple of the form (outputs, state), where:
        outputs: A list of the same length as decoder_inputs of 2D tensors with
        shape [batch_size x output_size] containing generated outputs.
        state: The state of each cell at the final time-step. 
            It is a 2D tensor of shape [batch_size x cell.state_size]
            (Note that in some cases, like basic RNN cell or GRU cell, outputs and 
            states can be the same. They are different for LSTM cells through.)
    """

    with tf.variable_scope(scope or "RNN_Decoder"):
        state = initial_state
        outputs = []
        prev = None
        for i, inp in enumerate(decoder_inputs):
            if loop_function is not None and prev is not None:
                with tf.variable_scope("loop_function", reuse = True):
                    inp = loop_function(prev, i)
            if i > 0:
                tf.get_variable_scope().reuse_variables()
            output, state = cell(inp, state)

            outputs.append(output)

            if loop_function is not None:
                prev = output

    return outputs, state

def embedding_attention_decoder(decoder_inputs, initial_state,attention_state,
                                cell, num_symbols, embedding_size, num_heads = 1,
                                output_size = None, output_projection = None,
                                feed_previous = False,
                                update_embedding_for_previous = True,
                                dtype = tf.float32, scope = None,
                                initial_state_attention = False, beam_search = True, beam_size = 10):
    """
    RNN decoder with embedding and attention and a pure-decoding option. 


    """

def embedding_attention_seq2seq(encoder_inputs, decoder_inputs, cell_1, cell_2,
                                num_encoder_symbols, num_decoder_symbols,
                                embedding_size,
                                num_heads = 1, output_projection = None,
                                feed_previous = False, dtype = tf.float32,
                                scope = None, initial_state_attention = False, beam_search = True, beam_size = 10):
    """
    Embedding sequence-to-sequence model with attention.

    This model first embeds encoder_inputs by a newly created embedding (of shape
    [num_encoder_symbols x input_size]). Then it runs an RNN to encode
    embedded encoder_inputs into a state vector. It keeps the outpus of this 
    RNN at every step to use for attention later. Next, it embeds decoder_inputs 
    by another newly created embedding (of shape [num_decoder_symbols x input_size]).
    Then it runs attention decoder, initialized with the last encoder state, on embedded 
    decoder_inputs and attending to encoder outputs.

    Args:
        encoder_inputs: A list of 1D int32 Tensors of shape [batch_size].
        decoder_inputs: A list of 1D int32 Tensors of shape [batch_size].
        cell: rnn_cell.RNNCell defining the cell function and size.
        num_encoder_symbols: Integer; number of symbols on the encoder side.
        num_decoder_symbols: Integer; number of symbols on the decoder side.
        embedding_size: Integer, the length of the embedding vector for each symbol.
        num_heads: Number of attention heads that read from attention_states.
        output_projection: None or a pair (W, B) of output projection weights and
        biases; W has shape [output_size x num_decoder_symbols] and B has
        shape [num_decoder_symbols]; if provided and feed_previous=True, each
        fed previous output will first be multiplied by W and added B.
        feed_previous: Boolean or scalar Boolean Tensor; if True, only the first
        of decoder_inputs will be used (the "GO" symbol), and all other decoder
        inputs will be taken from previous outputs (as in embedding_rnn_decoder).
        If False, decoder_inputs are used as given (the standard decoder case).
        dtype: The dtype of the initial RNN state (default: tf.float32).
        scope: VariableScope for the created subgraph; defaults to
        "embedding_attention_seq2seq". 
        initial_state_attention: If False (default), initial attentions are zero.
        If True, initialize the attentions from the initial state and attention
        states.
    Returns:
        A tuple of the form (outputs, state), where:
        outputs: A list of the same length as decoder_inputs of 2D Tensors with
            shape [batch_size x num_decoder_symbols] containing the generated
            outputs.
        state: The state of each decoder cell at the final time-step.
            It is a 2D Tensor of shape [batch_size x cell.state_size].
    """

    with tf.variable_scope(scope or "embedding_attention_seq2seq"):
        #Encoder(EmbeddingWrapper:指定されたセルに入力埋め込みを追加する演算子)
        encoder_cell = tf.contrib.rnn.EmbeddingWrapper(
            cell_1, embedding_classes = num_encoder_symbols,
            embedding_size = embedding_size #reuse = tf.get_variable_scope().reuse 
        )
        encoder_outputs, encoder_state = tf.nn.static_rnn(
            encoder_cell, encoder_inputs,dtype=dtype
        )
        print("##### embedding_attention_seq2seq scope: {}".format(encoder_cell))
        print("Symbols")
        print(num_encoder_symbols)
        print(num_decoder_symbols)

        #First calculate a concatenation of encoder outputs to put attention on.
        top_states = [tf.reshape(e, [-1, 1, cell_1.output_size])
                        for e in encoder_outputs]
        attention_state = tf.concat(values = top_states, axis = 1)
        print(attention_state)

        #Decoder
        output_size = None
        if output_projection is None:
            cell_2 = tf.contrib.rnn.OutputProjectionWrapper(cell_2,num_decoder_symbols)
            output_size = num_decoder_symbols
        return embedding_attention_decoder(decoder_inputs, encoder_state, attention_states, cell_2,
          num_decoder_symbols, embedding_size, num_heads=num_heads,
          output_size=output_size, output_projection=output_projection,
          feed_previous=feed_previous,
          initial_state_attention=initial_state_attention, beam_search=beam_search, beam_size=beam_size)
