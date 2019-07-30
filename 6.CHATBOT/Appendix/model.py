# -*- coding: utf-8 -*-
import tensorflow as tf
import sys
import numpy as np
from configs import DEFINES


def make_lstm_cell(mode, hiddenSize, index):
    cell = tf.nn.rnn_cell.BasicLSTMCell(hiddenSize, name="lstm" + str(index), state_is_tuple=False)
    if mode == tf.estimator.ModeKeys.TRAIN:
        cell = tf.contrib.rnn.DropoutWrapper(cell, state_keep_prob=DEFINES.dropout_width)
    return cell

# 에스티메이터 모델 부분이다.
# freatures : tf.data.Dataset.map을 통해서 만들어진 
# features = {"input": input, "length": length}
# labels : tf.data.Dataset.map을 통해서 만들어진 target
# mode는 에스티메이터 함수를 호출하면 에스티메이터 
# 프레임워크 모드의 값이 해당 부분이다.
# params : 에스티메이터를 구성할때 params 값들이다. 
# (params={ # 모델 쪽으로 파라메터 전달한다.)
def Model(features, labels, mode, params):
    TRAIN = mode == tf.estimator.ModeKeys.TRAIN
    EVAL = mode == tf.estimator.ModeKeys.EVAL
    PREDICT = mode == tf.estimator.ModeKeys.PREDICT

    # 미리 정의된  임베딩 사용 유무를 확인 한다.
    # 값이 True이면 임베딩을 해서 학습하고 False이면 
    # onehotencoding 처리 한다.
    if params['embedding'] == True:
        # 가중치 행렬에 대한 초기화 함수이다.
        # xavier (Xavier Glorot와 Yoshua Bengio (2010)
        # URL : http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
        initializer = tf.contrib.layers.xavier_initializer()
        # 인코딩 변수를 선언하고 값을 설정한다.
        embedding_encoder = tf.get_variable(name="embedding_encoder",  # 이름
                                           shape=[params['vocabulary_length'], params['embedding_size']],  # 모양
                                           dtype=tf.float32,  # 타입
                                           initializer=initializer,  # 초기화 값
                                           trainable=True)  # 학습 유무
    else:
        # tf.eye를 통해서 사전의 크기 만큼의 단위행렬 
        # 구조를 만든다.
        embedding_encoder = tf.eye(num_rows=params['vocabulary_length'], dtype=tf.float32)
        # 인코딩 변수를 선언하고 값을 설정한다.
        embedding_encoder = tf.get_variable(name="embedding_encoder",  # 이름
                                           initializer=embedding_encoder,  # 초기화 값
                                           trainable=False)  # 학습 유무

    # embedding_lookup을 통해서 features['input']의 인덱스를
    # 위에서 만든 embedding_encoder의 인덱스의 값으로 변경하여 
    # 임베딩된 디코딩 배치를 만든다.
    embedding_encoder_batch = tf.nn.embedding_lookup(params=embedding_encoder, ids=features['input'])

    # 미리 정의된  임베딩 사용 유무를 확인 한다.
    # 값이 True이면 임베딩을 해서 학습하고 False이면 
    # onehotencoding 처리 한다.
    if params['embedding'] == True:
        # 가중치 행렬에 대한 초기화 함수이다.
        # xavier (Xavier Glorot와 Yoshua Bengio (2010)
        # URL : http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
        initializer = tf.contrib.layers.xavier_initializer()
        # 디코딩 변수를 선언하고 값을 설정한다.
        embedding_decoder = tf.get_variable(name="embedding_decoder",  # 이름
                                           shape=[params['vocabulary_length'], params['embedding_size']],  # 모양
                                           dtype=tf.float32,  # 타입
                                           initializer=initializer,  # 초기화 값
                                           trainable=True)  # 학습 유무
    else:
        # tf.eye를 통해서 사전의 크기 만큼의 단위행렬 
        # 구조를 만든다.
        embedding_decoder = tf.eye(num_rows=params['vocabulary_length'], dtype=tf.float32)
        # 인코딩 변수를 선언하고 값을 설정한다.
        embedding_decoder = tf.get_variable(name='embedding_decoder',  # 이름
                                           initializer=embedding_decoder,  # 초기화 값
                                           trainable=False)  # 학습 유무

    # 변수 재사용을 위해서 reuse=.AUTO_REUSE를 사용하며 범위를
    # 정해주고 사용하기 위해 scope설정을 한다.
    # make_lstm_cell이 "cell"반복적으로 호출 되면서 재사용된다.
    with tf.variable_scope('encoder_scope', reuse=tf.AUTO_REUSE):
        # 값이 True이면 멀티레이어로 모델을 구성하고 False이면 
        # 단일레이어로 모델을 구성 한다.
        if params['multilayer'] == True:
            # layerSize 만큼  LSTMCell을  encoder_cell_list에 담는다.
            encoder_cell_list = [make_lstm_cell(mode, params['hidden_size'], i) for i in range(params['layer_size'])]
            # MUltiLayer RNN CEll에 encoder_cell_list를 넣어 멀티 레이어를 만든다.
            rnn_cell = tf.contrib.rnn.MultiRNNCell(encoder_cell_list, state_is_tuple=False)
        else:
            # 단층 LSTMLCell을 만든다.
            rnn_cell = make_lstm_cell(mode, params['hidden_size'], "")
        # rnn_cell에 의해 지정된 반복적인 신경망을 만든다.
        # encoder_outputs(RNN 출력 Tensor)[batch_size, 
        # max_time, cell.output_size]
        # encoder_states 최종 상태  [batch_size, cell.state_size]
        encoder_outputs, encoder_states = tf.nn.dynamic_rnn(cell=rnn_cell,  # RNN 셀
                                                              inputs=embedding_encoder_batch,  # 입력 값
                                                              dtype=tf.float32)  # 타입
        # 변수 재사용을 위해서 reuse=.AUTO_REUSE를 사용하며 범위를 정해주고
        # 사용하기 위해 scope설정을 한다.
        # make_lstm_cell이 "cell"반복적으로 호출 되면서 재사용된다.
    with tf.variable_scope('decoder_scope', reuse=tf.AUTO_REUSE):
        # 값이 True이면 멀티레이어로 모델을 구성하고 False이면 단일레이어로
        # 모델을 구성 한다.
        if params['multilayer'] == True:
            # layer_size 만큼  LSTMCell을  decoder_cell_list에 담는다.
            decoder_cell_list = [make_lstm_cell(mode, params['hidden_size'], i) for i in range(params['layer_size'])]
            # MUltiLayer RNN CEll에 decoder_cell_list를 넣어 멀티 레이어를 만든다.
            rnn_cell = tf.contrib.rnn.MultiRNNCell(decoder_cell_list, state_is_tuple=False)
        else:
            # 단층 LSTMLCell을 만든다.
            rnn_cell = make_lstm_cell(mode, params['hidden_size'], "")

        decoder_state = encoder_states
        # 매 타임 스텝에 나오는 아웃풋을 저장하는 리스트 두개를 만든다. 
        # 하나는 토큰 인덱스는 predict_tokens 저장
        # 다른 하나는 temp_logits에 logits 저장한다.
        predict_tokens = list()
        temp_logits = list()

        # 평가인 경우에는 teacher forcing이 되지 않도록 해야한다.
        # 따라서 학습이 아닌경우에 is_train을 False로 하여 teacher forcing이 되지 않도록 한다.
        output_token = tf.ones(shape=(tf.shape(encoder_outputs)[0],), dtype=tf.int32) * 1
        # 전체 문장 길이 만큼 타임 스텝을 돌도록 한다.
        for i in range(DEFINES.max_sequence_length):
            # 두 번쨰 스텝 이후에는 teacher forcing을 적용하는지 확률에 따라 결정하도록 한다.
            # teacher forcing rate은 teacher forcing을 어느정도 줄 것인지를 조절한다.
            if TRAIN:
                if i > 0:
                    # tf.cond를 통해 rnn에 입력할 입력 임베딩 벡터를 결정한다 여기서 true인 경우엔 입력된 output값 아닌경우에는 이전 스텝에
                    # 나온 output을 사용한다.
                    input_token_emb = tf.cond(
                        tf.logical_and( # 논리 and 연산자
                            True,
                            tf.random_uniform(shape=(), maxval=1) <= params['teacher_forcing_rate'] # 률에 따른 labels값 지원 유무
                        ),
                        lambda: tf.nn.embedding_lookup(embedding_decoder, labels[:, i-1]),  # labels 정답을 넣어주고 있다.
                        lambda: tf.nn.embedding_lookup(embedding_decoder, output_token) # 모델이 정답이라고 생각 하는 값
                    )
                else:
                    input_token_emb = tf.nn.embedding_lookup(embedding_decoder, output_token) # 모델이 정답이라고 생각 하는 값
            else: # 평가 및 예측은 여기를 진행해야 한다. 
                input_token_emb = tf.nn.embedding_lookup(embedding_decoder, output_token)

            # 어텐션 적용 부분
            if params['attention'] == True:
                W1 = tf.keras.layers.Dense(params['hidden_size'])
                W2 = tf.keras.layers.Dense(params['hidden_size'])
                V = tf.keras.layers.Dense(1)
                # (?, 256) -> (?, 128)
                hidden_with_time_axis = W2(decoder_state)
                # (?, 128) -> (?, 1, 128)
                hidden_with_time_axis = tf.expand_dims(hidden_with_time_axis, axis=1)
                # (?, 1, 128) -> (?, 25, 128)
                hidden_with_time_axis = tf.manip.tile(hidden_with_time_axis, [1, DEFINES.max_sequence_length, 1])
                # (?, 25, 1)
                score = V(tf.nn.tanh(W1(encoder_outputs) + hidden_with_time_axis))
                # score = V(tf.nn.tanh(W1(encoderOutputs) + tf.manip.tile(tf.expand_dims(W2(decoder_state), axis=1), [1, DEFINES.maxSequenceLength, 1])))
                # (?, 25, 1)
                attention_weights = tf.nn.softmax(score, axis=-1)
                # (?, 25, 128)
                context_vector = attention_weights * encoder_outputs
                # (?, 25, 128) -> (?, 128)
                context_vector = tf.reduce_sum(context_vector, axis=1)
                # (?, 256)
                input_token_emb = tf.concat([context_vector, input_token_emb], axis=-1)

            # RNNCell을 호출하여 RNN 스텝 연산을 진행하도록 한다.
            input_token_emb = tf.keras.layers.Dropout(0.5)(input_token_emb)
            decoder_outputs, decoder_state = rnn_cell(input_token_emb, decoder_state)
            decoder_outputs = tf.keras.layers.Dropout(0.5)(decoder_outputs)
            # feedforward를 거쳐 output에 대한 logit값을 구한다.
            output_logits = tf.layers.dense(decoder_outputs, params['vocabulary_length'], activation=None)

            # softmax를 통해 단어에 대한 예측 probability를 구한다.
            output_probs = tf.nn.softmax(output_logits)
            output_token = tf.argmax(output_probs, axis=-1)

            # 한 스텝에 나온 토큰과 logit 결과를 저장해둔다.
            predict_tokens.append(output_token)
            temp_logits.append(output_logits)

        # 저장했던 토큰과 logit 리스트를 stack을 통해 메트릭스로 만들어 준다.
        # 만들게 뙤면 차원이 [시퀀스 X 배치 X 단어 feature 수] 이렇게 되는데
        # 이를 transpose하여 [배치 X 시퀀스 X 단어 feature 수] 로 맞춰준다.
        predict = tf.transpose(tf.stack(predict_tokens, axis=0), [1, 0])
        logits = tf.transpose(tf.stack(temp_logits, axis=0), [1, 0, 2])

        print(predict.shape)
        print(logits.shape)

    if PREDICT:
        if params['serving'] == True:
            export_outputs = {
                'indexs': tf.estimator.export.PredictOutput(predict) # 서빙 결과값을 준다.
            }

        predictions = {  # 예측 값들이 여기에 딕셔너리 형태로 담긴다.
            'indexs': predict,  # 시퀀스 마다 예측한 값
            'logits': logits,  # 마지막 결과 값
        }
        # 에스티메이터에서 리턴하는 값은 tf.estimator.EstimatorSpec 
        # 객체를 리턴 한다.
        # mode : 에스티메이터가 수행하는 mode (tf.estimator.ModeKeys.PREDICT)
        # predictions : 예측 값
        if params['serving'] == True:
            return tf.estimator.EstimatorSpec(mode, predictions=predictions, export_outputs=export_outputs)

        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # 마지막 결과 값과 정답 값을 비교하는 
    # tf.nn.sparse_softmax_cross_entropy_with_logits(로스함수)를 
    # 통과 시켜 틀린 만큼의
    # 에러 값을 가져 오고 이것들은 차원 축소를 통해 단일 텐서 값을 반환 한다.
    # pad의 loss값을 무력화 시킨다. pad가 아닌값은 1 pad인 값은 0을 주어 동작
    # 하도록 한다.
    # 정답 차원 변경을 한다. [배치 * max_sequence_length * vocabulary_length]  
    # logits과 같은 차원을 만들기 위함이다.
    labels_ = tf.one_hot(labels, params['vocabulary_length'])
    
    if TRAIN and params['loss_mask'] == True:
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels_))
        masks = features['length']

        loss = loss * tf.cast(masks, tf.float32)
        loss = tf.reduce_mean(loss)
    else:
       loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels_))
    # 라벨과 결과가 일치하는지 빈도 계산을 통해 
    # 정확도를 측정하는 방법이다.
    accuracy = tf.metrics.accuracy(labels=labels, predictions=predict, name='accOp')

    # 정확도를 전체값으로 나눈 값이다.
    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])

    # 평가 mode 확인 부분이며 평가는 여기 까지 
    # 수행하고 리턴한다.
    if EVAL:
        # 에스티메이터에서 리턴하는 값은 
        # tf.estimator.EstimatorSpec 객체를 리턴 한다.
        # mode : 에스티메이터가 수행하는 mode (tf.estimator.ModeKeys.EVAL)
        # loss : 에러 값
        # eval_metric_ops : 정확도 값
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

    # 파이썬 assert구문으로 거짓일 경우 프로그램이 종료 된다.
    # 수행 mode(tf.estimator.ModeKeys.TRAIN)가 
    # 아닌 경우는 여기 까지 오면 안되도록 방어적 코드를 넣은것이다.
    assert TRAIN

    # 아담 옵티마이저를 사용한다.
    optimizer = tf.train.AdamOptimizer(learning_rate=DEFINES.learning_rate)
    # 에러값을 옵티마이저를 사용해서 최소화 시킨다.
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    # 에스티메이터에서 리턴하는 값은 tf.estimator.EstimatorSpec 객체를 리턴 한다.
    # mode : 에스티메이터가 수행하는 mode (tf.estimator.ModeKeys.EVAL)
    # loss : 에러 값
    # train_op : 그라디언트 반환
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
