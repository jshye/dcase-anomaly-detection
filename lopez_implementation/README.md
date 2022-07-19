
* config.yaml
    
    * dataset directory, log directory 설정
    * hyperparameter 설정
    * 각 machine type 별 batch size, mel spectrogram parameter 설정 

* train.py => dev_data 사용
    
    * 각 machine type 별 model 만들어서 학습
    * -m, --machine_type
    
        ToyCar, ToyTrain, fan, gearbox, pump, slider, valve 중에서 0개 이상 선택

        ** case-sensitive **
        ```
        python train.py -m ToyCar  // run training for ToyCar only
        python train.py            // run training for all machine types
        ```
    * accuracy: section id 분류 정확도
    * anomaly_score: DCASE baseline의 공식 사용, prob: section id class의 softmax probability
    * decision_threshold: anomaly score의 gamma distribution으로부터 구함

* test.py => dev_data 사용

    * normal/anomaly 분류 정확도 평가
    * --ep: decision threshold 구할 때 사용할 score distribution의 epoch 설정
    
        (model save된 epoch으로 설정하는 것이 일반적)
    
    * 학습된 모델 load해서 machine type, section id 별 metric 구한 후 csv 생성
    * AUC, pAUC, precision, recall, F1 score
    * 전체 domain, section에 대한 harmonic & arithmetic mean score
    * 전체 machine type, domain, section에 대한 harmonic & arithmetic mean score
    
    ```
    # example
    python test.py -m ToyCar --ep 40
    ```