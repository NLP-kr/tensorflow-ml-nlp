# -*- coding: utf-8 -*-
import tensorflow as tf
import sys

from configs import DEFINES
import numpy as np


def layer_norm(inputs, eps=1e-6):
    # LayerNorm(x + Sublayer(x))
    feature_shape = inputs.get_shape()[-1:]
    #  평균과 표준편차을 넘겨 준다.
    mean = tf.keras.backend.mean(inputs, [-1], keepdims=True)
    std = tf.keras.backend.std(inputs, [-1], keepdims=True)
    beta = tf.get_variable("beta", initializer=tf.zeros(feature_shape))
    gamma = tf.get_variable("gamma", initializer=tf.ones(feature_shape))

    return gamma * (inputs - mean) / (std + eps) + beta


def sublayer_connection(inputs, sublayer, dropout=0.2):
    # LayerNorm(x + Sublayer(x))
    outputs = layer_norm(inputs + tf.keras.layers.Dropout(dropout)(sublayer))
    return outputs


def feed_forward(inputs, num_units):
    # FFN(x) = max(0, xW1 + b1)W2 + b2
    feature_shape = inputs.get_shape()[-1]
    inner_linear_weight = tf.get_variable('inner_dense', shape=[1, feature_shape, num_units])
    outer_linear_weight = tf.get_variable('outer_dense', shape=[1, num_units, feature_shape])

    inner_layer = tf.nn.conv1d(inputs, inner_linear_weight, stride=1, padding='VALID')
    inner_layer = tf.nn.relu(inner_layer)
    outputs = tf.nn.conv1d(inner_layer, outer_linear_weight, stride=1, padding='VALID')

    return outputs


def positional_encoding(dim, sentence_length):
    # Positional Encoding
    # paper: https://arxiv.org/abs/1706.03762
    # P E(pos,2i) = sin(pos/100002i/dmodel)
    # P E(pos,2i+1) = cos(pos/100002i/dmodel)
    encoded_vec = np.array([pos / np.power(10000, 2 * i / dim)
                            for pos in range(sentence_length) for i in range(dim)])
    encoded_vec[::2] = np.sin(encoded_vec[::2])
    encoded_vec[1::2] = np.cos(encoded_vec[1::2])
    return tf.constant(encoded_vec.reshape([sentence_length, dim]), dtype=tf.float32)


def scaled_dot_product_attention(query, key, value, masked=False):
    # Attention(Q, K, V ) = softmax(QKt / root dk)V
    key_dim_size = float(key.get_shape().as_list()[-1])
    key = tf.transpose(key, perm=[0, 2, 1])
    outputs = tf.matmul(query, key) / tf.sqrt(key_dim_size)

    if masked:
        diag_vals = tf.ones_like(outputs[0, :, :])
        tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()
        masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])

        paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(masks, 0), paddings, outputs)

    attention_map = tf.nn.softmax(outputs)

    return tf.matmul(attention_map, value)


def multi_head_attention(query, key, value, num_units, heads, masked=False):
    feature_shape = query.get_shape()[-1]

    query_linear_weight = tf.get_variable('query_dense', shape=[1, feature_shape, num_units])
    key_linear_weight = tf.get_variable('key_dense', shape=[1, feature_shape, num_units])
    value_linear_weight = tf.get_variable('value_dense', shape=[1, feature_shape, num_units])
    attn_linear_weight = tf.get_variable('attn_dense', shape=[1, num_units, feature_shape])

    query = tf.nn.conv1d(query, query_linear_weight, stride=1, padding='SAME')
    key = tf.nn.conv1d(key, key_linear_weight, stride=1, padding='SAME')
    value = tf.nn.conv1d(value, value_linear_weight, stride=1, padding='SAME')

    query = tf.concat(tf.split(query, heads, axis=-1), axis=0)
    key = tf.concat(tf.split(key, heads, axis=-1), axis=0)
    value = tf.concat(tf.split(value, heads, axis=-1), axis=0)

    attention_map = scaled_dot_product_attention(query, key, value, masked)

    attn_outputs = tf.concat(tf.split(attention_map, heads, axis=0), axis=-1)
    attn_outputs = tf.nn.conv1d(attn_outputs, attn_linear_weight, stride=1, padding='VALID')

    return attn_outputs


def encoder_module(inputs, model_dim, ffn_dim, heads):
    with tf.variable_scope('self_attention'):
        self_attn = sublayer_connection(inputs, multi_head_attention(inputs, inputs, inputs,
                                                                     model_dim, heads))
    with tf.variable_scope('feed_forward'):
        outputs = sublayer_connection(self_attn, feed_forward(self_attn, ffn_dim))
    return outputs


def decoder_module(inputs, encoder_outputs, model_dim, ffn_dim, heads):
    with tf.variable_scope('self_attention'):
        masked_self_attn = sublayer_connection(inputs, multi_head_attention(inputs, inputs, inputs,
                                                                            model_dim, heads, masked=True))
    with tf.variable_scope('enc_dec_attention'):
        self_attn = sublayer_connection(masked_self_attn, multi_head_attention(masked_self_attn, encoder_outputs,
                                                                               encoder_outputs, model_dim, heads))
    with tf.variable_scope('feed_forward'):
        outputs = sublayer_connection(self_attn, feed_forward(self_attn, ffn_dim))

    return outputs


def encoder(inputs, model_dim, ffn_dim, heads, num_layers):
    outputs = inputs
    for i in range(num_layers):
        with tf.variable_scope('encoder_layer_' + str(i + 1)):
            outputs = encoder_module(outputs, model_dim, ffn_dim, heads)

    return outputs


def decoder(inputs, encoder_outputs, model_dim, ffn_dim, heads, num_layers):
    outputs = inputs
    for i in range(num_layers):
        with tf.variable_scope('decoder_layer_' + str(i+1)):
            outputs = decoder_module(outputs, encoder_outputs, model_dim, ffn_dim, heads)

    return outputs


def Model(features, labels, mode, params):
    TRAIN = mode == tf.estimator.ModeKeys.TRAIN
    EVAL = mode == tf.estimator.ModeKeys.EVAL
    PREDICT = mode == tf.estimator.ModeKeys.PREDICT

    position_encode = positional_encoding(params['embedding_size'], params['max_sequence_length'])

    if params['xavier_initializer']:
        embedding_initializer = 'glorot_normal'
    else:
        embedding_initializer = 'uniform'

    embedding = tf.keras.layers.Embedding(params['vocabulary_length'],
                                          params['embedding_size'],
                                          embeddings_initializer=embedding_initializer)

    with tf.variable_scope('encoder', reuse=False):
        x_embedded_matrix = embedding(features['input']) + position_encode
        encoder_outputs = encoder(x_embedded_matrix, params['model_hidden_size'], params['ffn_hidden_size'],
                                      params['attention_head_size'], params['layer_size'])

    loop_count = params['max_sequence_length'] if PREDICT else 1

    for i in range(loop_count):
        reuse = True if i > 0 else False
        if i > 0:
            output = tf.concat([tf.ones((output.shape[0], 1), dtype=tf.int64), predict[:, :-1]], axis=-1)
        else:
            output = features['output']

        with tf.variable_scope('decoder', reuse=reuse):
            y_embedded_matrix = embedding(output) + position_encode
            decoder_outputs = decoder(y_embedded_matrix, encoder_outputs, params['model_hidden_size'],
                                      params['ffn_hidden_size'],
                                      params['attention_head_size'], params['layer_size'])

            logits = tf.layers.dense(decoder_outputs, params['vocabulary_length'], use_bias=False)
            predict = tf.argmax(logits, 2)

    if PREDICT:
        predictions = {
            'indexs': predict,
            'logits': logits
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # 정답 차원 변경을 한다. [배치 * max_sequence_length * vocabulary_length]  
    # logits과 같은 차원을 만들기 위함이다.
    labels_ = tf.one_hot(labels, params['vocabulary_length'])
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels_))

    accuracy = tf.metrics.accuracy(labels=labels, predictions=predict)

    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])

    if EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

    assert TRAIN

    # lrate = d−0.5 *  model · min(step_num−0.5, step_num · warmup_steps−1.5)
    optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
