import tensorflow as tf
import data
import sys
import model as ml

from configs import DEFINES
	
if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    arg_length = len(sys.argv)
    
    if(arg_length < 2):
        raise Exception("Don't call us. We'll call you")
  
    
    # 데이터를 통한 사전 구성 한다.
    char2idx,  idx2char, vocabulary_length = data.load_vocabulary()

    # 테스트용 데이터 만드는 부분이다.
    # 인코딩 부분 만든다.
    input = ""
    for i in sys.argv[1:]:
        input += i 
        input += " "
        
    print(input)
    predic_input_enc, predic_input_enc_length = data.enc_processing([input], char2idx)
    # 학습 과정이 아니므로 디코딩 입력은 
    # 존재하지 않는다.(구조를 맞추기 위해 넣는다.)
    # 학습 과정이 아니므로 디코딩 출력 부분도 
    # 존재하지 않는다.(구조를 맞추기 위해 넣는다.)
    predic_target_dec, _ = data.dec_target_processing([""], char2idx)

    if DEFINES.serving == True:
        # 모델이 저장된 위치를 넣어 준다.  export_dir
        predictor_fn = tf.contrib.predictor.from_saved_model(
            export_dir="/home/evo_mind/DeepLearning/NLP/Work/ChatBot2_Final/data_out/model/1541575161"
        )
    else:
        # 에스티메이터 구성한다.
        classifier = tf.estimator.Estimator(
                model_fn=ml.Model, # 모델 등록한다.
                model_dir=DEFINES.check_point_path, # 체크포인트 위치 등록한다.
                params={ # 모델 쪽으로 파라메터 전달한다.
                    'hidden_size': DEFINES.hidden_size,  # 가중치 크기 설정한다.
                    'layer_size': DEFINES.layer_size,  # 멀티 레이어 층 개수를 설정한다.
                    'learning_rate': DEFINES.learning_rate,  # 학습율 설정한다.
                    'teacher_forcing_rate': DEFINES.teacher_forcing_rate, # 학습시 디코더 인풋 정답 지원율 설정
                    'vocabulary_length': vocabulary_length,  # 딕셔너리 크기를 설정한다.
                    'embedding_size': DEFINES.embedding_size,  # 임베딩 크기를 설정한다.
                    'embedding': DEFINES.embedding,  # 임베딩 사용 유무를 설정한다.
                    'multilayer': DEFINES.multilayer,  # 멀티 레이어 사용 유무를 설정한다.
                    'attention': DEFINES.attention, #  어텐션 지원 유무를 설정한다.
                    'teacher_forcing': DEFINES.teacher_forcing, # 학습시 디코더 인풋 정답 지원 유무 설정한다.
                    'loss_mask': DEFINES.loss_mask, # PAD에 대한 마스크를 통한 loss를 제한 한다.
                    'serving': DEFINES.serving # 모델 저장 및 serving 유무를 설정한다.
                })

    if DEFINES.serving == True:
        predictions = predictor_fn({'input':predic_input_enc, 'output':predic_target_dec})
        data.pred2string(predictions, idx2char)
    else:
        # 예측을 하는 부분이다.
        predictions = classifier.predict(
            input_fn=lambda:data.eval_input_fn(predic_input_enc, predic_target_dec, DEFINES.batch_size))
        # 예측한 값을 인지 할 수 있도록 
        # 텍스트로 변경하는 부분이다.
        data.pred2string(predictions, idx2char)
