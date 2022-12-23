#  DPLAN

#### "Toward Deep Supervised Anomaly Detection: Reinforcement Learning from Partially Labeled Anomaly Data” 논문의 DPLAN 강화학습 이상감지 모델을 구현하여 ADFA 데이터에 적용

# Environment
* scikit-learn==0.24.2
* gym==0.18.0
* pandas==1.2.4
* python==3.8.8

# Dataset 
* ADFA   
  * 리눅스 혹은 윈도우 호스트에서 수집된 데이터셋
  * 호스트 기반 침입탐지시스템을 평가하기 위해 공개된 데이터

# DPLAN Model Framework
![image](https://user-images.githubusercontent.com/121276658/209277044-16bb3bbf-7e46-4483-9d2d-fda2befbbd2d.png)
* ### Anomaly detection Agent   
  * 이상감지를 수행하면서 라벨링이 되어있는 소수의 비정상 데이터를 통해 이상치 탐지의 정확도를 향상  
  * 라벨링이 되어 있지 않는 비정상 데이터도 판별하고 감지할 수 있도록 제한 없이 진행  
 
* ### Anomaly-biased simulation Environment   
  * Agent가 새로운 형태의 이상치를 감지할 수 있도록 라벨링이 되지 않은 데이터 exploration을 지원
 
* ### Combined Reward function   
  * 라벨링이 되거나 되지않은 이상치로부터 supervisory 정보를 사용해서 exploration-exploitation의 균형을 달성

# Experiment
* To change hyperparameter : config.py  
* To run: python main.py
