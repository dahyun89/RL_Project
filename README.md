#  DPLAN : Deep Reinforcement Learning from Partially Labeled Anomaly Data

#### 2021년 Guansong Pang의 "Toward Deep Supervised Anomaly Detection: Reinforcement Learning from Partially Labeled Anomaly Data” 논문의 DPLAN 강화학습 이상감지 모델을 기반으로 네트워크 이상탐지 데이터에 적용 및 구현

### DPLAN 모델 프레임워크
![image](https://user-images.githubusercontent.com/121276658/209277044-16bb3bbf-7e46-4483-9d2d-fda2befbbd2d.png)

* #### Anomaly detection Agent   
 * 이상감지를 수행하면서 라벨링이 된 적은 비정상 데이터를 통해 이상치 탐지의 정확도를 향상  
 * 라벨링이 되어 있지 않는 비정상 데이터도 판별하고 감지할 수 있도록 제한 없이 진행  
 * 에이전트는 라벨링이 되고 되어있지 않은 데이터를 통한  simulated environment와 상호작용하며 이상 감지를 수행
 
* #### Anomaly-biased simulation Environment   
 * 에이전트가 새로운 형태의 이상치를 감지할 수 있도록 라벨링이 되지 않은 데이터 탐색을 지원
 
* #### Combined Reward function   
 * 라벨링이 되고 되지 않은 이상치로부터 supervisory 정보를 사용해서 exploration-exploitation의 균형을 달성
