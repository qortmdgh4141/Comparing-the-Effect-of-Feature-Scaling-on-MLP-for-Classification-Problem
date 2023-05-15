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
  from keras import initializers
  from keras.utils import np_utils
  from keras.datasets import fashion_mnist

  from keras.optimizers import Adam
  from keras.models import Sequential
  from keras.layers import Flatten, Dense, Dropout, Input, Conv2D, MaxPooling2D
  from sklearn.model_selection import train_test_split


  import numpy as np
  import pandas as pd
  import seaborn as sns
  import matplotlib.pyplot as plt
  ```

- _**Data Preparation**_ <br/> 
  
  ```
  # 학습용, 검증용, 테스트용으로 분리하여 MNIST 데이터 셋트 로딩
  (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
  x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, random_state=42)

  # 학습용 & 검즘용 & 테스트용 데이터의 차원
  print(f"학습용 데이터의 차원 : 입력 데이터 {x_train.shape} / 라벨 데이터 / {y_train.shape}") 
  print(f"검증용 데이터의 차원 : 입력 데이터 {x_val.shape} / 라벨 데이터 / {y_val.shape}")
  print(f"테스트용 데이터의 차원 : 입력 데이터 {x_test.shape} / 라벨 데이터 / {y_test.shape}")
  ```
  
  ```
  # 10개의 이미지와 목표 변수를 그래프로 출력
  plt.figure(figsize=(12, 2))
  for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(x_train[i], cmap='gray')
    plt.title(str(y_train[i]))
    plt.axis('off')
  plt.show()
  ```
  
  <img src="https://github.com/qortmdgh4141/Comparing-Performance-of-MLP-and-CNN-for-Classification-Problem/blob/main/image/image_label_graph.png?raw=true">
  
- _**Exploratory Data Analysis (EDA)**_ <br/> 
  
  ```
  # 입력 데이터의 차원 변환 : 3차원(이미지 수, 28, 28) -> 2차원 (이미지 수, 784)
  x_train_reshaped = x_train.reshape(x_train.shape[0], 784)

  # 데이터 프레임으로 변형하여 널 값의 빈도 확인
  x_train_df = pd.DataFrame(x_train_reshaped)
  total_null_count = x_train_df.isnull().sum().sum()
  print(f"널값의 개수 : {total_null_count}개")
  ```

  ```
  # 각 열별로 픽셀의 강도 분석
  x_train_df.describe()
  ```
  
  ```
  # 목표변수의 라벨별 빈도 계산 후 데이터 프레임으로 변환
  y_cnt = pd.DataFrame(y_train).value_counts()
  df = pd.DataFrame(y_cnt, columns=['Count'])

  # 인덱스 리셋 및 문자열로 변환
  df.reset_index(inplace=True)  
  df['Label'] = df[0].astype(str)

  # 컬러맵 설정 및 바차트 생성
  cmap = plt.cm.Set3 
  fig, ax = plt.subplots(figsize=(12, 3)) 
  bars = ax.bar(df['Label'], df['Count'], color=cmap(np.arange(len(df))))

  # 바 위에 라벨 갯수 출력
  for i, count in enumerate(df['Count']):
      ax.text(i, count + 100, str(count), ha='center', fontsize=7)

  # 그래프 레이블과 제목 설정 및  y축 범위 늘리기 (현재 최댓값의 110%로 범위 지정)
  ax.set_xlabel('Label')
  ax.set_ylabel('Frequency')
  ax.set_title('Label Counts')
  ax.set_ylim(0, df['Count'].max() * 1.1)

  plt.show() # 그래프 출력
  ```
  
  <img src="https://github.com/qortmdgh4141/Comparing-Performance-of-MLP-and-CNN-for-Classification-Problem/blob/main/image/vertical_bar_graph.png?raw=true">
    
- _**Feature Scaling**_ <br/>  
  
  ```
  # 입력데이터는 모두 0~255 사이 값이기 때문에 각각 255로 나누어 0~1로 정규화
  x_train = x_train.astype('float32') / 255
  x_val = x_val.astype('float32') / 255
  x_test = x_test.astype('float32') / 255  
  ```
  
- _**One-Hot Encoding**_ <br/> 
  
  ```
  # 라벨 데이터의 원-핫 인코딩
  y_train = np_utils.to_categorical(y_train)
  y_val = np_utils.to_categorical(y_val)
  y_test = np_utils.to_categorical(y_test)
  <br/> 

### 4. &nbsp; Training and Testing MLP Model <br/><br/>

- _Optimized MLP Model_

  ```
  """
  1. 완전연결 계층 (Dense Layer)
      - 노드 수: 512 or 10
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
  mlp_model = Sequential()
  mlp_model.add(Flatten(input_shape=(28, 28)))
  mlp_model.add(Dropout(0.5))
  mlp_model.add(Dense(512, activation='relu', kernel_initializer=initializers.HeNormal()))
  mlp_model.add(Dense(10, activation='softmax'))

  mlp_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy']) 

  mlp_model.summary() # 모형 구조 출력 
  ```

  ```
  # 학습
  results_mlp = mlp_model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=100, batch_size=128)
  ```
    
  ```
  # 학습된 모형 테스트 
  mlp_score = mlp_model.evaluate(x_test, y_test)
  mlp_accuracy = round(mlp_score[1]*100, 2)
  print(f"MLP 모델 기반 테스트 데이터의 손실함수 값 : {round(mlp_score[0], 2)}")
  print(f"MLP 모델 기반 테스트 데이터의 정확도      : {mlp_accuracy}%")
  ```
  
  ```
  # 학습된 모형으로 테스트 데이터를 예측
  mlp_y_pred = mlp_model.predict(x_test)

  # 예측 값과 실제 값의 라벨
  mlp_y_pred_class = np.argmax(mlp_y_pred, axis=1)
  y_test_class = np.argmax(y_test, axis=1)

  # 교차표 : 실제 값 대비 예측 값 (주대각원소의 값이 정확하게 분류된 빈도, 그 외는 오분류 빈도)
  mlp_crosstab = pd.crosstab(y_test_class,mlp_y_pred_class)
  mlp_crosstab
  ```
  <br/> 
  
### 5. &nbsp; Training and Testing CNN Model <br/><br/>

- _Optimized MLP Model_

  ```
  """
  1. 합성곱 층 (Convolutional Layer)
      - 필터 개수: 32 or 64, 커널 크기 : (3, 3)
      - 합성곱 층은 입력 데이터에 대해 필터(커널)를 이용하여 지역적인 특징을 추출 특성 맵(Feature Map)을 생성

  2. 풀링 층 (Pooling Layer) 
      - 최대 풀링 크기: (2, 2)
      - 풀링 층은 공간적인 불변성을 제공하고, 특성 맵의 크기를 줄여 계산량을 감소시키고, 추상화된 특징을 더 강조함

  3. 완전연결 계층 (Dense Layer)
      - 노드 수: 512 or 10
      - 완전연결 계층은 모든 입력과 출력을 연결하는 전통적인 신경망 계층
      - 추상적인 특징을 학습하고, 다양한 클래스에 대한 확률 분포를 출력하는 역할을 수행

  4. 드롭아웃(Dropout) 층
      - 신경망의 학습 과정에서 과적합을 줄이기 위해 사용되는 정규화 기법인 드롭아웃(Dropout) 층을 추가
      - 드롭아웃은 학습 과정 중에 신경망의 일부 유닛(neuron)을 임의로 선택하여 비활성화시킴으로써,
        모델이 특정 유닛에 과도하게 의존하는 것을 방지하거 일반화 능력을 향상

  5. 은닉층의 활성화 함수 :  Relu
     - 입력값이 0보다 작을 경우는 0으로 출력하고, 0보다 큰 경우는 그대로 출력하는 비선형 함수인 Relu 함수로 설정
     - ReLU 활성화 함수를 사용할 때, 가중치 초기화에 따른 그래디언트 소실 문제를 완화하기 위해 은닉층의 가중치는 He 초깃값을 사용

  6. 출력층의 활성화 함수 :  Softmax
     - 주로 다중 클래스 분류 문제에서 출력층에서 사용되는 활성화 함수인  Softmax로 설정
     - Softmax 함수는 입력받은 값을 정규화하여 각 클래스에 속할 확률을 계산하며, 모든 클래스에 대한 확률의 합은 1

  7. 최적화 알고리즘 : Adam
     - Momentum과 RMSProp의 장점을 결합한 최적화 알고리즘인 Adam(Adaptive Moment Estimation)을 사용
     - Momentum은 : 기울기의 방향을 고려하여 학습 속도를 조절 
     - RMSProp : 기울기 크기를 고려하여 학습 속도를 조절

  8. 손실 함수 : Cross-Entropy Loss Function
     - 출력층에서 Softmax 함수를 사용할 경우, 손실 함수로는 주로 크로스 엔트로피 손실 함수를 사용
     - 크로스 엔트로피 손실 함수(Cross-Entropy Loss Function)는 실제 타깃 값에 해당하는 클래스에 대해서만 오차를 계산하며, 
       오차를 최소화하는 방향으로 학습이 진행

  9. 정확도 평가 지표 : Accuracy
     - 분류 모델의 성능을 평가하는 지표 중 하나인 Accuracy를 사용
     - 예측한 클래스가 실제 타깃 클래스와 일치하는 경우를 정확한 분류로 간주하고, 이를 전체 샘플 수로 나누어 정확도를 계산

  10. 배치 사이즈 / 학습 반복 횟수 / 학습률 : 128 / 100 / 0.001

  # 모형 구조
  cnn_model = Sequential()
  cnn_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1), kernel_initializer=initializers.HeNormal()))
  cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
  cnn_model.add(Dropout(0.5))
  cnn_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1), kernel_initializer=initializers.HeNormal()))
  cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
  cnn_model.add(Dropout(0.5))

  cnn_model.add(Flatten())
  cnn_model.add(Dense(512, activation='relu', kernel_initializer=initializers.HeNormal()))
  cnn_model.add(Dropout(0.5))
  cnn_model.add(Dense(10, activation='softmax'))

  cnn_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])

  cnn_model.summary() # 모형 구조 출력 
  ```

  ```
  # 학습
  results_cnn = cnn_model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=100, batch_size=128)
  ```
    
  ```
  # 학습된 모형 테스트 
  cnn_score = cnn_model.evaluate(x_test, y_test)
  cnn_accuracy = round(cnn_score[1]*100, 2)
  print(f"CNN 모델 기반 테스트 데이터의 손실함수 값 : {round(cnn_score[0], 2)}")
  print(f"CNN 모델 기반 테스트 데이터의 정확도      : {cnn_accuracy}%")
  ```
  
  ```
  # 학습된 모형으로 테스트 데이터를 예측
  cnn_y_pred = cnn_model.predict(x_test)

  # 예측 값과 실제 값의 라벨
  cnn_y_pred_class = np.argmax(cnn_y_pred, axis=1)
  y_test_class = np.argmax(y_test, axis=1)

  # 교차표 : 실제 값 대비 예측 값 (주대각원소의 값이 정확하게 분류된 빈도, 그 외는 오분류 빈도)
  cnn_crosstab = pd.crosstab(y_test_class, cnn_y_pred_class)
  cnn_crosstab
  ```
  <br/> 
  
### 5. &nbsp; Research Results  <br/><br/>
    
- _The objective of this study was to train MLP-based and CNN-based models using the Fashion-MNIST dataset and compare their classification performance._ <br/> 

- _The graph below shows the change in both Loss and Accuracy with increasing Epochs during the training process of both the MLP-based and CNN-based models._ <br/> <br/> 
  
  ```
  def plot_loss_and_accuracy(train_loss, val_loss, train_acc, val_acc, model_name):
      epochs = range(1, len(train_loss) + 1)

      plt.figure(figsize=(12, 6))

      # Loss 그래프
      plt.subplot(2, 2, 1)
      plt.plot(epochs, train_loss, 'b', label='Training Loss')
      plt.plot(epochs, val_loss, 'r', label='Validation Loss')
      plt.title(f'{model_name} Model - Training and Validation Loss')
      plt.xlabel('Epoch')
      plt.ylabel('Loss')
      plt.legend()

      # Accuracy 그래프
      plt.subplot(2, 2, 2)
      plt.plot(epochs, train_acc, 'b', label='Training Accuracy')
      plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
      plt.title(f'{model_name} Model - Training and Validation Accuracy')
      plt.xlabel('Epoch')
      plt.ylabel('Accuracy')
      plt.legend()

      plt.tight_layout()

      plt.show()

  # MLP 모델 결과 그래프 출력
  plot_loss_and_accuracy(results_mlp.history['loss'], results_mlp.history['val_loss'],
                         results_mlp.history['accuracy'], results_mlp.history['val_accuracy'], 'MLP')

  # CNN 모델 결과 그래프 출력
  plot_loss_and_accuracy(results_cnn.history['loss'], results_cnn.history['val_loss'],
                         results_cnn.history['accuracy'], results_cnn.history['val_accuracy'], 'CNN')
  ```
  
  <img src="https://github.com/qortmdgh4141/Comparing-Performance-of-MLP-and-CNN-for-Classification-Problem/blob/main/image/line_graph.png?raw=true">

- _In the MLP model, a slight overfitting phenomenon was observed. This indicates that the MLP model might be overly fitted to the training data, leading to a decrease in generalization performance on new data._ <br/>
 
- _In contrast, examining the change in Loss in the graph for the CNN model shows no occurrence of overfitting. This suggests that the CNN model effectively learns the local structure and features of images compared to the MLP model, demonstrating better generalization capability on new data._ <br/>

- _Furthermore, the CNN model exhibited a very fast convergence rate and high accuracy. Due to its specialized architecture for image processing, the CNN model was able to better capture the visual features of images and was more suitable for the classification task, surpassing the convergence rate and accuracy of the MLP model._ <br/><br/> 

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
          plt.text(w + 0.7, y + h / 2.0 + 0.015, "{}".format(int(w)), fontsize=8, ha='left', va='center')

  # MLP 모델 및 CNN 모델의 오분류 빈도
  mlp_error_count = len(y_test_class) - np.sum(y_test_class == mlp_y_pred_class)
  cnn_error_count = len(y_test_class) - np.sum(y_test_class == cnn_y_pred_class)
  error_counts = [mlp_error_count, cnn_error_count]

  # 막대 그래프로 오분류 빈도 표현
  models = ['MLP', 'CNN']
  cmap_list = ['Reds', 'Blues']

  fig, ax = plt.subplots(figsize=(12, 4))
  bars = ax.barh(models, error_counts, color='white', alpha=0.7)
  gradientbars(bars, cmap_list)

  ax.set_ylabel('Model', fontsize=12)
  ax.set_xlabel('Error Count', fontsize=12)
  ax.set_title('< Error Count Comparison between MLP and CNN >', fontsize=10)

  plt.show()

  ```
  
  <img src="https://github.com/qortmdgh4141/Comparing-Performance-of-MLP-and-CNN-for-Classification-Problem/blob/main/image/horizontal_bar_graph.png?raw=true">
  
- _Next, comparing the misclassification frequencies of the MLP and CNN models on the actual test data, the MLP model had 544 misclassifications, while the CNN model had 385 misclassifications. This indicates that the CNN model correctly predicted about 200 more answers compared to the MLP model._ <br/>
 
- _This result clearly demonstrates the performance difference between MLP and CNN models, proving that CNN models are more suitable and show superior performance in fashion item image classification tasks._ <br/>

- _Additionally, despite its complex model structure, the CNN model did not experience overfitting and demonstrated better generalization capability. This indicates that the possibility of overfitting in deep learning training is not determined solely by the number of training parameters, but also by the model structure._<br/> <br/> <br/>
 
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
