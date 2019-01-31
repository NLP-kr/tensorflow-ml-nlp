# ChatBot(seq2seq)
NLG를 활용한 ChatBot을 개발함  

![images](images/seq2seq.png)  


# Learning Data
Title|Contents|Other
--|--|--
데이터 이름|Chatbot data
데이터 용도|한국어 챗봇  학습을 목적으로 사용한다.
데이터 권한|MIT 라이센스
데이터 출처|https://github.com/songys/Chatbot_data (송영숙님)

# Requirement
Python 3.6.6   
tensorflow 1.11   
konlpy   
pandas   
sklearn   

# Project Structure
    .
    ├── data_in                     # 데이터가 존재하는 영역
        ├── ChatBotData.csv         # 전체 데이터
        ├── ChatBotData.csv_short   # 축소된 데이터 (테스트 용도)
        ├── README.md               # 데이터 저자 READMD 파일
    ├── data_out                    # 출력 되는 모든 데이터가 모이는 영역
        ├── vocabularyData.voc      # 사전 파일
        ├── check_point             # check_point 저장 공간
    ├── configs.py                  # 모델 설정에 관한 소스
    ├── data.py                     # data 전처리 및 모델에 주입되는 data set 만드는 소스
    ├── main.py                     # 전체적인 프로그램이 시작되는 소스
    ├── model.py                    # 모델이 들어 있는 소스
    └── predict.py                  # 학습된 모델로 실행 해보는 소스    
   

# Config
```
tf.app.flags.DEFINE_integer('batchSize', 100, 'batch size') # 배치 크기  
tf.app.flags.DEFINE_integer('trainSteps', 10000, 'train steps') # 학습 에포크  
tf.app.flags.DEFINE_float('dropoutWidth', 0.5, 'dropout width') # 드롭아웃 크기  
tf.app.flags.DEFINE_integer('layerSize', 3, 'layer size') # 멀티 레이어 크기 (multi rnn)  
tf.app.flags.DEFINE_integer('hiddenSize', 128, 'weights size') # 가중치 크기  
tf.app.flags.DEFINE_float('learningRate', 1e-3, 'learning rate') # 학습률  
tf.app.flags.DEFINE_string('dataPath', './data/ChatBotData.csv', 'data path') #  데이터 위치  
tf.app.flags.DEFINE_string('vocabularyPath', './data/vocabularyData.voc', 'vocabulary path') # 사전 위치  
tf.app.flags.DEFINE_string('checkPointPath', './checkPoint', 'check point path') # 체크 포인트 위치  
tf.app.flags.DEFINE_integer('shuffleSeek', 1000, 'shuffle random seek') # 셔플 시드값  
tf.app.flags.DEFINE_integer('maxSequenceLength', 25, 'max sequence length') # 시퀀스 길이  
tf.app.flags.DEFINE_integer('embeddingSize', 128, 'embedding size') # 임베딩 크기  
tf.app.flags.DEFINE_boolean('tokenizeAsMorph', True, 'set morph tokenize') # 형태소에 따른 토크나이징 사용 유무  
tf.app.flags.DEFINE_boolean('embedding', True, 'Use Embedding flag') # 임베딩 유무 설정  
tf.app.flags.DEFINE_boolean('multilayer', True, 'Use Multi RNN Cell') # 멀티 RNN 유무  
```

# Usage
python main.py

# Predict
python predict.py 남자친구가 너무 잘 생겼어

# Reference
Title|Contents
--|--
Data|[Chatbot data](https://github.com/songys/Chatbot_data)  
Paper|[Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/pdf/1406.1078.pdf)  
Paper|[Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473.pdf)

<!--
# Author
ChangWookJun / @changwookjun (changwookjun@gmail.com)
-->
