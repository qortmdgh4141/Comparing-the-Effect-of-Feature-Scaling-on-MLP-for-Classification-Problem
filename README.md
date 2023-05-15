# 📉  Comparing the Effect of Feature Scaling on MLP for Classification Problem
<br/>
  
### 1. &nbsp; Research Objective <br/><br/>

- _The objective of this study is to construct and train an MLP classification model using the "Wine Recognition Dataset" from the UCI Machine Learning Repository, which includes outliers. To achieve this, I will experiment with two feature scaling techniques, normalization and standardization, to determine which one has a more positive impact on the performance of the model._ <br/>

- _Generally, it is known that normalization technique performs better than standardization technique in improving the performance of a model when dealing with datasets that contain outliers. However, for datasets with outliers removed, a normalization technique may improve the performance of the model, or a standardization technique may improve the performance of the model, depending on a variety of factors, including the structure of the model and the characteristics of the dataset._ <br/>

- _Based on this information, I conducted the following experiment:_  <br/>

  - _First, I performed outlier detection and removal using the Interquartile Range (IQR) method on the "Wine Dataset."_ <br/>
  
  - _Then, I applied normalization and standardization techniques to the dataset with the outliers removed._ <br/>
  
  - _Finally, I used the two transformed datasets to train MLP models with the same architecture, respectively, and compared the performance of the models._ <br/>
  
- _Through this experiment, I was able to understand the impact of outlier removal and feature scaling techniques on model performance from different perspectives, and I expect that this information can be utilized in future research and practical applications._ <br/><br/><br/> 

### 2. &nbsp; Key Components of the Neural Network Model and Experimental Settings  <br/><br/>

- _Dense Layer_ <br/>

  - _Number of nodes: (256 or 3)._ <br/>

  - _The dense layer is a traditional neural network layer that connects all inputs and outputs. It learns abstract features and outputs probability distribution for various classes._<br/><br/>
  
- _Dropout Layer_ <br/>

  - _The dropout layer is one of the regularization techniques used to reduce overfitting during the neural network training process._ <br/>

  - _Dropout randomly deactivates some units (neurons) of the neural network during training, preventing the model from relying too heavily on specific units and improving generalization capability._<br/><br/>

- _Activation function for hidden layers : ReLU Function_ <br/>

  - _The ReLU function is a non-linear function that outputs 0 for negative input values and keeps the output as is for positive input values._ <br/>

  - _To alleviate the issue of gradient vanishing caused by weight initialization when using ReLU activation function, the weights of the hidden layers were initialized using He initialization._<br/><br/>

- _Activation function for the output layer : Softmax Function_ <br/>

  - _The softmax function is commonly used as the activation function for the output layer in multi-class classification problems._ <br/>

  - _The softmax function normalizes the input values to calculate the probability of belonging to each class, and the sum of probabilities for all classes is 1._<br/><br/>

- _Optimization Algorithm : Adam (Adaptive Moment Estimation)_ <br/>

  - _The Adam optimization algorithm, which combines the advantages of Momentum, which adjusts the learning rate considering the direction of gradients, and RMSProp, which adjusts the learning rate considering the magnitude of gradients, was used._ <br/>

  - _The softmax function normalizes the input values to calculate the probability of belonging to each class, and the sum of probabilities for all classes is 1._<br/><br/>

- _Loss Function : Cross-Entropy Loss Function_ <br/>

  - _When using the softmax function in the output layer, the cross-entropy loss function is commonly used as the loss function._ <br/>

  - _The cross-entropy loss function calculates the error only for the classes corresponding to the actual target values and updates the model in the direction of minimizing the error._<br/><br/>

- _Evaluation Metric : Accuracy_ <br/>

  - _Accuracy is one of the evaluation metrics used to assess the performance of a classification model._ <br/>

  - _Accuracy considers the prediction as correct if it matches the actual target class and calculates it by dividing it by the total number of samples._<br/><br/>

- _Batch Size & Maximum Number of Learning Iterations_ <br/>

  - _In this experiment, the batch size is 128, and the model is trained by iterating up to a maximum of 100 times._<br/>
  
  - _The number of batch size and iterations during training affects the speed and accuracy of the model, and I, as the researcher conducting the experiment, have set the number of batch size and iterations based on my experience of tuning deep learning models._<br/><br/> <br/> 

### 3. &nbsp; Data Preprocessing and Analysis <br/><br/>

- _**Package Settings**_ <br/> 
  
  ```
  from sklearn.datasets import load_wine
  from sklearn.model_selection import train_test_split
  from sklearn.preprocessing import MinMaxScaler, StandardScaler

  from keras import initializers
  from keras.utils import np_utils
  from keras.optimizers import Adam
  from keras.models import Sequential
  from keras.layers import Flatten, Dense, Dropout

  import random
  import numpy as np
  import pandas as pd
  import seaborn as sns
  import matplotlib.pyplot as plt
  ```

- _**Data Preparation**_ <br/> 
  
  ```
  # 데이터 불러오기
  data = load_wine(as_frame = True)

  # 데이터 세트 개요 : 각 열별 최대 및 최소 분포를 파악
  print(data.DESCR)
  ```
  
  ```
  # 입력 데이터와 라벨 데이터를 출력함으로써 개략적인 수치를 확인 
  x_df = data['data']
  y_df = pd.DataFrame(data['target'], columns=['target'])
  all_data_df = pd.concat([x_df, y_df], axis=1)
  all_data_df
  ```
  
  ```
  # 데이터 프레임의 특성(feature) 이름과 목표 변수(target) 이름 출력
  print("[ 데이터 프레임의 특성(feature) ]")
  print("\n".join(f" - {feature}" for feature in data.feature_names))

  # 목표 변수에서 순서대로 'class_0'는 0, 'class_1'은 1, 'class_2'은 2을 의미
  print("\n[ 데이터 프레임의 목표 변수(target) ]")
  print("\n".join(f" - {target}" for target in data.target_names))
  ```
  
  ```
  # data 부분을 8:1:1 비율로 학습용과 검증용, 테스트용 데이터셋으로 분리
  x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=723)
  x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, random_state=723)
  ```
  
- _**Exploratory Data Analysis (EDA)**_ <br/> 
  
  ```
  # 이상치 제거 함수 정의
  def remove_outliers_iqr(df, threshold=1.5):
      columns = df.columns
      for column in columns:
          q1 = df[column].quantile(0.25)
          q3 = df[column].quantile(0.75)
          iqr = q3 - q1
          lower_bound = q1 - threshold * iqr
          upper_bound = q3 + threshold * iqr
          df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
      return df

  # 이상치 제거 및 비율 계산
  filtered_x_train = remove_outliers_iqr(x_train)
  filtered_y_train = y_train[filtered_x_train.index]
  outlier_ratio = (len(x_train) - len(filtered_x_train)) / len(x_train) * 100

  # 결과 출력
  print("이상치 제거 전 데이터프레임 크기:", x_train.shape)
  print("이상치 제거 후 데이터프레임 크기:", filtered_x_train.shape)
  print("이상치 비율: {:.2f}%".format(outlier_ratio))
  ```
  
  ```
  # 데이터 시각화를 위해 표준화
  std_1 = StandardScaler()
  x_train_std= std_1.fit_transform(x_train)
  std_2 = StandardScaler()
  filtered_x_train_std = std_2.fit_transform(filtered_x_train)

  # 그래프 크기 및 스타일, 폰트 설정
  plt.figure(figsize=(14, 4), dpi=100)
  plt.style.use('seaborn-darkgrid')
  plt.rcParams.update({'font.size': 12})

  # 그래프 타이틀, 축 레이블 및 범위 설정
  plt.subplot(1, 2, 1)
  sns.boxplot(data=x_train_std, palette='Set3')
  plt.title("Box Plot of Scaled Features", fontsize=14)
  plt.xlabel("Features", fontsize=10)
  plt.ylabel("Scaled Values", fontsize=10)
  plt.ylim(-4, 4)

  plt.subplot(1, 2, 2)
  sns.boxplot(data=filtered_x_train_std, palette='Set3')
  plt.title(f"\nBox Plot of Scaled Features (Outliers Removed)", fontsize=14)
  plt.xlabel("Features", fontsize=10)
  plt.ylabel("Scaled Values", fontsize=10)
  plt.ylim(-4, 4)  

  plt.tight_layout()

  plt.show()
  ```
  
  <img src="https://github.com/qortmdgh4141/Comparing-the-Effect-of-Feature-Scaling-on-MLP-for-Classification-Problem/blob/main/image/boxplot_graph.png?raw=true">
    
- _**Feature Scaling**_ <br/>  
  
  ```
  x_train = filtered_x_train
  y_train = filtered_y_train

  # 피처 스케일링 : 최소-최대 정규화 
  mm = MinMaxScaler()
  x_train_mm = mm.fit_transform(x_train)
  x_val_mm = mm.fit_transform(x_val)
  x_test_mm = mm.fit_transform(x_test)

  # 피처 스케일링 : 표준화
  std = StandardScaler()
  x_train_std = std.fit_transform(x_train)
  x_val_std = std.fit_transform(x_val)
  x_test_std = std.fit_transform(x_test)

  # 라벨 데이터의 원-핫 인코딩
  y_train = np_utils.to_categorical(y_train)
  y_val = np_utils.to_categorical(y_val)
  y_test = np_utils.to_categorical(y_test)
  ```

### 4. &nbsp; Training and Testing MLP Model <br/><br/>

- _Optimized MLP Model_

  ```
  """
  1. 완전연결 계층 (Dense Layer)
      - 노드 수: 256 or 3
      - 완전연결 계층은 모든 입력과 출력을 연결하는 전통적인 신경망 계층
      - 추상적인 특징을 학습하고, 다양한 클래스에 대한 확률 분포를 출력하는 역할을 수행

  2. 드롭아웃(Dropout) 층
      - 신경망의 학습 과정에서 과적합을 줄이기 위해 사용되는 정규화 기법인 드롭아웃(Dropout) 층을 추가
      - 드롭아웃은 학습 과정 중에 신경망의 일부 유닛(neuron)을 임의로 선택하여 비활성화시킴으로써,
        모델이 특정 유닛에 과도하게 의존하는 것을 방지하거 일반화 능력을 향상

  3. 은닉층의 활성화 함수 :  Relu
     - 입력값이 0보다 작을 경우는 0으로 출력하고, 0보다 큰 경우는 그대로 출력하는 비선형 함수인 Relu 함수로 설정
     - ReLU 활성화 함수를 사용할 때, 가중치 초기화에 따른 그래디언트 소실 문제를 완화하기 위해 은닉층의 가중치는 He 초깃값을 사용

  4. 출력층의 활성화 함수 :  Softmax
     - 주로 다중 클래스 분류 문제에서 출력층에서 사용되는 활성화 함수인  Softmax로 설정
     - Softmax 함수는 입력받은 값을 정규화하여 각 클래스에 속할 확률을 계산하며, 모든 클래스에 대한 확률의 합은 1

  5. 최적화 알고리즘 : Adam
     - Momentum과 RMSProp의 장점을 결합한 최적화 알고리즘인 Adam(Adaptive Moment Estimation)을 사용
     - Momentum은 : 기울기의 방향을 고려하여 학습 속도를 조절 
     - RMSProp : 기울기 크기를 고려하여 학습 속도를 조절

  6. 손실 함수 : Cross-Entropy Loss Function
     - 출력층에서 Softmax 함수를 사용할 경우, 손실 함수로는 주로 크로스 엔트로피 손실 함수를 사용
     - 크로스 엔트로피 손실 함수(Cross-Entropy Loss Function)는 실제 타깃 값에 해당하는 클래스에 대해서만 오차를 계산하며, 
       오차를 최소화하는 방향으로 학습이 진행

  7. 정확도 평가 지표 : Accuracy
     - 분류 모델의 성능을 평가하는 지표 중 하나인 Accuracy를 사용
     - 예측한 클래스가 실제 타깃 클래스와 일치하는 경우를 정확한 분류로 간주하고, 이를 전체 샘플 수로 나누어 정확도를 계산

  8. 배치 사이즈 / 학습 반복 횟수 / 학습률 : 128 / 100 / 0.001
  """
  # 모형 구조  
  model = Sequential()
  model.add(Flatten(input_shape=(13,)))
  model.add(Dropout(0.25))
  model.add(Dense(256, activation='relu', kernel_initializer=initializers.HeNormal()))
  model.add(Dense(3, activation='softmax'))

  model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy']) 

  model.summary() # 모형 구조 출력 
  ```

  ```
  # 학습
  results_mm = model.fit(x_train_mm, y_train, validation_data=(x_val_mm, y_val), epochs=100, batch_size=128)
  ```
    
  ```
  # 학습된 모형 테스트 
  score_mm = model.evaluate(x_test_mm, y_test)
  accuracy_mm = round(score_mm[1]*100, 2)
  print(f"정규화 기반 MLP 모델의 테스트 데이터에 대한 손실함수 값 : {round(score_mm[0], 2)}")
  print(f"정규화 기반 MLP 모델의 테스트 데이터에 대한 정확도      : {accuracy_mm}%")
  ```
  
  ```
  # 학습된 모형으로 테스트 데이터를 예측
  y_pred_mm = model.predict(x_test_mm)

  # 예측 값과 실제 값의 라벨
  y_pred_class_mm = np.argmax(y_pred_mm, axis=1)
  y_test_class = np.argmax(y_test, axis=1)

  # 교차표 : 실제 값 대비 예측 값 (주대각원소의 값이 정확하게 분류된 빈도, 그 외는 오분류 빈도)
  crosstab_mm = pd.crosstab(y_test_class ,y_pred_class_mm)
  crosstab_mm
  ```
  
  ```
  # 학습
  results_std = model.fit(x_train_std, y_train, validation_data=(x_val_std, y_val), epochs=100, batch_size=128)
  ```
  
  ```
  # 학습된 모형 테스트 
  score_std = model.evaluate(x_test_std, y_test)
  accuracy_std = round(score_std[1]*100, 2)
  print(f"표준화 기반 MLP 모델의 테스트 데이터에 대한 손실함수 값 : {round(score_std[0], 2)}")
  print(f"표준화 기반 MLP 모델의 테스트 데이터에 대한 정확도      : {accuracy_std}%"))
  ```
  
  ```
  # 학습된 모형으로 테스트 데이터를 예측
  y_pred_std = model.predict(x_test_std)

  # 예측 값과 실제 값의 라벨
  y_pred_class_std = np.argmax(y_pred_std, axis=1)
  y_test_class = np.argmax(y_test, axis=1)

  # 교차표 : 실제 값 대비 예측 값 (주대각원소의 값이 정확하게 분류된 빈도, 그 외는 오분류 빈도)
  crosstab_std = pd.crosstab(y_test_class ,y_pred_class_std)
  crosstab_std
  ```
  <br/> 
    
### 5. &nbsp; Research Results  <br/><br/>
    
- _In this study, I performed IQR-based outlier detection and removal on the 'Wine Recognition Dataset' from the UCI Machine Learning Repository, which contains outliers, and trained MLP models with the same structure using two datasets with normalization and standardization._ <br/> 

- _The graph below illustrates the changes in loss and accuaracy as the number of epochs increases during the training process of the MLP models._ <br/> <br/> 
  
  ```
  def plot_loss_and_accuracy(train_loss, val_loss, train_acc, val_acc, model_name):
      epochs = range(1, len(train_loss) + 1)

      plt.figure(figsize=(14, 6))

      # Loss 및 Accuracy 그래프
      plt.subplot(2, 2, 1)
      plt.plot(epochs, train_loss, 'b', label='Training Loss')
      plt.plot(epochs, val_loss, 'r', label='Validation Loss')
      plt.title(f'{model_name}-based MLP - Training and Validation Loss', fontsize=13)
      plt.xlabel('Epoch')
      plt.ylabel('Loss')
      plt.ylim(0, 1.7)
      plt.legend()

      plt.subplot(2, 2, 2)
      plt.plot(epochs, train_acc, 'b', label='Training Accuracy')
      plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
      plt.title(f'{model_name}-based MLP  - Training and Validation Accuracy', fontsize=13)
      plt.xlabel('Epoch')
      plt.ylabel('Accuracy')
      plt.ylim(0, 1.0)
      plt.legend()

      plt.tight_layout()

      plt.show()

  # 그래프 출력
  plot_loss_and_accuracy(results_mm.history['loss'], results_mm.history['val_loss'],
                         results_mm.history['accuracy'], results_mm.history['val_accuracy'], 'Normalization')
  plot_loss_and_accuracy(results_std.history['loss'], results_std.history['val_loss'],
                         results_std.history['accuracy'], results_std.history['val_accuracy'], 'Standardization')
  ```
  
  <img src="https://github.com/qortmdgh4141/Comparing-the-Effect-of-Feature-Scaling-on-MLP-for-Classification-Problem/blob/main/image/line_graph.png?raw=true">

- _As shown in the graph, the MLP model trained with normalization suffered from underfitting during the training process. Underfitting occurs when the model fails to sufficiently train from the training data, leading to an inadequate understanding of the relationship between the training data and new input data._ <br/>
 
- _On the other hand, the MLP model trained with standardization shows rapid convergence of the loss to the optimal value as the number of epochs increases. Moreover, the decreasing Loss for both the training and validation data suggests that the model has high generalization ability._ <br/>
  
- _Furthermore, examining the changes in accuracy for the training and validation datasets, the MLP model trained with standardization demonstrates a gradual increase in accuracy. However, the model trained with normalization maintains a low accuracy due to underfitting._ <br/><br/>

  ```
  def gradientbars(bars, cmap_list):
      grad = np.atleast_2d(np.linspace(0, 1, 256)).T
      ax = bars[0].axes
      lim = ax.get_xlim() + ax.get_ylim()
      ax.axis(lim)
      max_width = max([bar.get_width() for bar in bars])
      for i, bar in enumerate(bars):
          bar.set_facecolor("none")
          x, y = bar.get_xy()
          w, h = bar.get_width(), bar.get_height()
          ax.imshow(grad, extent=[x, x + w, y, y + h], aspect="auto", cmap=cmap_list[i])
          plt.text(w + 0.7, y + h / 2.0 + 0.015, "{}%".format(int(w)), fontsize=12, ha='left', va='center')

  acc = [accuracy_mm, accuracy_std]

  models = ['Normalization', 'Standardization']
  cmap_list = ['Reds', 'Blues']

  fig, ax = plt.subplots(figsize=(12, 4))
  bars = ax.barh(models, acc, color='white', alpha=0.7)
  gradientbars(bars, cmap_list)

  ax.set_ylabel('Feature Scaling Method\n', fontsize=14)
  ax.set_xlabel('Accuracy', fontsize=14)
  ax.set_title('< Accuracy Comparison : Standardization vs Normalization >\n', fontsize=14)

  plt.show()
  ```
  
  <img src="https://github.com/qortmdgh4141/Comparing-the-Effect-of-Feature-Scaling-on-MLP-for-Classification-Problem/blob/main/image/bar_graph.png?raw=true">
  
- _Lastly, the graph below illustrates the accuracy of the MLP models trained with normalization and standardization when applied to the test dataset. The model trained with standardization has a high accuracy of about 94%, which can be considered a strong classification model. However, the accuracy of the model trained with normalization is about 33%, suggesting that it is not performing well in terms of classification._ <br/>
 
- _In conclusion, the performance difference between the two models is very large, about 3 times, indicating that the feature scaling technique has a very significant impact on the performance of the model. Therefore, in future research or practical applications, feature scaling should be considered as an important preprocessing step along with outlier removal, which will make a great contribution to improving model performance._ <br/> <br/> <br/>
 
--------------------------
### 💻 S/W Development Environment
<p>
  <img src="https://img.shields.io/badge/Windows 10-0078D6?style=flat-square&logo=Windows&logoColor=white"/>
  <img src="https://img.shields.io/badge/Google Colab-black?style=flat-square&logo=Google Colab&logoColor=yellow"/>
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=Python&logoColor=white"/>
</p>
<p>
  <img src="https://img.shields.io/badge/Keras-D00000?style=flat-square&logo=keras&logoColor=white"/>
  <img src="https://img.shields.io/badge/scikit learn-blue?style=flat-square&logo=scikitlearn&logoColor=F7931E"/>
  <img src="https://img.shields.io/badge/Numpy-013243?style=flat-square&logo=Numpy&logoColor=blue"/>
</p>

### 🚀 Machine Learning Model
<p>
  <img src="https://img.shields.io/badge/MLP-5C5543?style=flat-square?"/>
  <img src="https://img.shields.io/badge/CNN-4169E1?style=flat-square?"/>
</p> 

### 💾 Datasets used in the project
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Fashion-MNIST Dataset <br/>
