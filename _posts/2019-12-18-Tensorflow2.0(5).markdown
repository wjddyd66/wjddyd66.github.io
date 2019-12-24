---
layout: post
title:  "Save and load models"
date:   2019-12-19 09:00:20 +0700
categories: [Tnesorflow2.0]
---
<script type="text/javascript" src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS_HTML"></script>
### Save and load models
Tensorflow 2.0에서는 Keras 사용을 권장하고 사용하게 된다.  
이번 Post에서는 실제로 Training된 Model을 Save하고 Load하는 방법에 대해서 다룬다.  
기본적으로 <a href="https://wjddyd66.github.io/categories/#keras">Keras Category</a>에서 Model을 저장하고 불러오는 방법과 <a href="https://wjddyd66.github.io/keras/Keras(5)/#%EC%BC%80%EB%9D%BC%EC%8A%A4-%EC%BD%9C%EB%B0%B1%EA%B3%BC-%ED%85%90%EC%84%9C%EB%B3%B4%EB%93%9C%EB%A5%BC-%EC%82%AC%EC%9A%A9%ED%95%9C-%EB%94%A5%EB%9F%AC%EB%8B%9D-%EB%AA%A8%EB%8D%B8-%EA%B2%80%EC%82%AC%EC%99%80-%EB%AA%A8%EB%8B%88%ED%84%B0%EB%A7%81">Keras Callback</a>에서 Keras의 Callback에 대한 사전지식이 있으면 수월하게 넘어갈 수 있는 Post이다.  

사전 사항으로서 pyyaml, h5py 2개의 Python Package를 설치하여야 한다.  
<br>

#### 필요한 Library Import
```python
from __future__ import absolute_import, division, print_function, unicode_literals

import os

import tensorflow as tf
from tensorflow import keras

print(tf.version.VERSION)
```
<br>
2.0.0  
<br><br>

#### Get an example dataset
Keras의 <code>tf.keras.datasets.mnist.load_data()</code>를 활용하여 Mnist Dataset을 다운받는다.

```python
# Mnist Dataset Download
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Mnist Dataset Indexing
train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

# Dataset Normalization
train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0
```
<br>
<br><br>

#### Define a model
실제로 Save and Load할 Base Model을 선언한다.

```python
# Define a simple sequential model
def create_model():
    model = tf.keras.models.Sequential([
        keras.layers.Dense(512, activation='relu', input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

# Create a basic model instance
model = create_model()

# Display the model's architecture
model.summary()
```
<br>
```code
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense (Dense)                (None, 512)               401920    
_________________________________________________________________
dropout (Dropout)            (None, 512)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 10)                5130      
=================================================================
Total params: 407,050
Trainable params: 407,050
Non-trainable params: 0
_________________________________________________________________
```
<br>
<br><br>

#### Save checkpoints during training
Keras의 Callback을 사용하여 Model이 Training되는 동안 Checkpoints를 저장한다.  
Keras의 Callback에 대한 사전 지신은 링크를 참조하자. <a href="https://wjddyd66.github.io/keras/Keras(5)/#%EC%BC%80%EB%9D%BC%EC%8A%A4-%EC%BD%9C%EB%B0%B1%EA%B3%BC-%ED%85%90%EC%84%9C%EB%B3%B4%EB%93%9C%EB%A5%BC-%EC%82%AC%EC%9A%A9%ED%95%9C-%EB%94%A5%EB%9F%AC%EB%8B%9D-%EB%AA%A8%EB%8D%B8-%EA%B2%80%EC%82%AC%EC%99%80-%EB%AA%A8%EB%8B%88%ED%84%B0%EB%A7%81">Keras Callback</a>  
Keras의 Callback 중 <code>tf.keras.callbacks.ModelCheckpoint()</code>를 사용한다.  

**tf.keras.callbacks.ModelCheckpoint() Argument**  
- filepath: Model file을 저장할 경로
- monitor: Monitor할 수량
- verbose: 0 or 1 Training되는 동안 상황을 지켜볼 것인지 아닌지
- save_best_only: 가장 성능이 좋은 Model File만 저장한다.
- save_weights_only: True이면 모델의 가중치만 저장한다.

위의 Argument를 제외하고 많은 Option을 제공한다. 다양한 Option과 자세한 사용법은 링크를 참조하자.  
참조: <a href="https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint?version=stable">tf.keras.callbacks.ModelCheckpoint() 사용법</a>  

아래 Code를 살펴보면 다음과 같다.  
<code>tf.keras.callbacks.ModelCheckpoint()</code>: Keras Callback Object 선언
- filepath: Model File이 저장될 경로
- verbose = 1: Training 중 매 Epoch마다 확인


```python
# Keras CallBack Modelcheckpoint 의 Option 설정
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Keras CallBack 선언
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,verbose=1)

# Train the model with the new callback
model.fit(train_images, 
          train_labels,  
          epochs=10,
          validation_data=(test_images,test_labels),
          callbacks=[cp_callback])  # Pass callback to training
```
<br>
```code
Train on 1000 samples, validate on 1000 samples
Epoch 1/10
 832/1000 [=======================>......] - ETA: 0s - loss: 0.0011 - accuracy: 1.0000    
Epoch 00001: saving model to training_1/cp.ckpt

...

Epoch 00010: saving model to training_1/cp.ckpt
INFO:tensorflow:Assets written to: training_1/cp.ckpt/assets
1000/1000 [==============================] - 0s 390us/sample - loss: 8.3009e-04 - accuracy: 1.0000 - val_loss: 0.5114 - val_accuracy: 0.8790
```
<br>
위의 Directory의 결과를 확인하면 다음과 같다.
```python
!ls {checkpoint_dir}
```
<br>
checkpoint  cp.ckpt  cp.ckpt.data-00000-of-00001  cp.ckpt.index  
Directory에 저장되는 File의 의미는 다음과 같다.  
- data file: it is TensorBundle collection, save the values of all variables.
- index file: it is a string-string immutable table(tensorflow::table::Table). Each key is a name of a tensor and its value is a serialized BundleEntryProto. Each BundleEntryProto describes the metadata of a tensor: which of the "data" files contains the content of a tensor, the offset into that file, checksum, some auxiliary data, etc.

<br><br>

#### Load Model
아래 Code는 Weight가 Training되지 않은 Model과 위에서 Training된 Model을 Load하여 Accuracy를 비교하는 Code이다.

```python
# Training되지 않은 Model Accuacy 측정
model = create_model()
loss, acc = model.evaluate(test_images,  test_labels, verbose=2)
print("Untrained model, accuracy: {:5.2f}%".format(100*acc))

# Training된 Model Accuacy 측정
# checkpoint_path = "training_1/cp.ckpt"
model.load_weights(checkpoint_path)
loss, acc = model.evaluate(test_images,  test_labels, verbose=2)
print("Untrained model, accuracy: {:5.2f}%".format(100*acc))
```
<br>
```code
1000/1 - 0s - loss: 2.2989 - accuracy: 0.0780
Untrained model, accuracy:  7.80%
1000/1 - 0s - loss: 0.4418 - accuracy: 0.8670
Untrained model, accuracy: 86.70%
```
<br>
<br><br>

#### Checkpoint callback options
아래 Code는 Checkpoint callback options를 추가적으로사용한다.  
- <code>checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"</code>: Model File의 저장을 str.format() 형태로서 정의할 수 있다. 
- <code>period=5</code>: Checkpoint callback은 5번의 Epoch마다 수행된다.
- <code>tf.train.latest_checkpoint()</code>: 마지막으로 저장된 Model File을 확인할 수 있다.

```python
# Include the epoch in the file name (uses `str.format`)
checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights every 5 epochs
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, 
    save_weights_only=True,
    period=5)

# Create a new model instance
model = create_model()

# Save the weights using the `checkpoint_path` format
model.save_weights(checkpoint_path.format(epoch=0))

# Train the model with the new callback
model.fit(train_images, 
              train_labels,
              epochs=50, 
              callbacks=[cp_callback],
              validation_data=(test_images,test_labels),
              verbose=0)

# Check the File
!ls {checkpoint_dir}

# Check Latest Model File
latest = tf.train.latest_checkpoint(checkpoint_dir)
print(latest)
```
<br>
```code
Epoch 00005: saving model to training_2/cp-0005.ckpt

Epoch 00010: saving model to training_2/cp-0010.ckpt

Epoch 00015: saving model to training_2/cp-0015.ckpt

Epoch 00020: saving model to training_2/cp-0020.ckpt

Epoch 00025: saving model to training_2/cp-0025.ckpt

Epoch 00030: saving model to training_2/cp-0030.ckpt

Epoch 00035: saving model to training_2/cp-0035.ckpt

Epoch 00040: saving model to training_2/cp-0040.ckpt

Epoch 00045: saving model to training_2/cp-0045.ckpt

Epoch 00050: saving model to training_2/cp-0050.ckpt
checkpoint			  cp-0025.ckpt.index
cp-0000.ckpt.data-00000-of-00001  cp-0030.ckpt.data-00000-of-00001
cp-0000.ckpt.index		  cp-0030.ckpt.index
cp-0005.ckpt.data-00000-of-00001  cp-0035.ckpt.data-00000-of-00001
cp-0005.ckpt.index		  cp-0035.ckpt.index
cp-0010.ckpt.data-00000-of-00001  cp-0040.ckpt.data-00000-of-00001
cp-0010.ckpt.index		  cp-0040.ckpt.index
cp-0015.ckpt.data-00000-of-00001  cp-0045.ckpt.data-00000-of-00001
cp-0015.ckpt.index		  cp-0045.ckpt.index
cp-0020.ckpt.data-00000-of-00001  cp-0050.ckpt.data-00000-of-00001
cp-0020.ckpt.index		  cp-0050.ckpt.index
cp-0025.ckpt.data-00000-of-00001
training_2/cp-0050.ckpt
```
<br>
<br><br>

#### Load Model
가장 마지막까지 Training된 Model을 Load하여 Accuracy를 확인하는 Code이다.

```python
# Create a new model instance
model = create_model()

# Load the previously saved weights
model.load_weights(latest)

# Re-evaluate the model
loss, acc = model.evaluate(test_images,  test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))
```
<br>
1000/1 - 0s - loss: 0.5970 - accuracy: 0.8750  
Restored model, accuracy: 87.50%  
<br><br>

#### Manually save weights
Keras의 CallBack을 사용하지 않고 저장하는 방법이다.  
<code>model.save_weights()</code>로서 저장한다.

```python
# Save the weights
path = './checkpoints/my_checkpoint'
model.save_weights(path)

# Create a new model instance
model = create_model()

# Restore the weights
model.load_weights(path)

# Evaluate the model
loss,acc = model.evaluate(test_images,  test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))
```
<br>
1000/1 - 0s - loss: 0.4418 - accuracy: 0.8670  
Restored model, accuracy: 86.70%  
<br><br>

#### Save and Load the entire model
위의 결과인 Check Points는 Model의 Weights들을 저장한 File이다.  
따라서 Model에 불러오거나, 적용한 Model에서 평가 및 새롭게 Training이 가능하다.  
하지만 Tensorflow Model처럼 Graph의 구조로서 이루워진 것이 아니기 때문에 File자체 만으로는 Model을 만들 수 없다.  

따라서 위에서는 다음과 같은 과정을 거쳤다.  
```python
# Model 선언
model = create_model()

# Model 가중치 적용
model.load_weights(path)
```
위와 같은 과정이아니라 Model자체를 저장하는 방법에 대해서 알아본다.  
Keras에서는 h5 Format을 사용하여 Tensorflow 1.x에서는 .pb 로서 정의하였다.  

먼저 Keras에서 제공하는 h5 Format으로서 Model을 정의한다.  
이러한 h5 Format으로서 저장하는 것은 다음과 같은 내용을 포함한다.  
- The weight values: .ckpt File 같은 Weights 정보
- The model's configuratrion: Graph의 정보
- The optimizer configuration


```python
# Create and train a new model instance.
model = create_model()
model.fit(train_images, train_labels, epochs=5)

# Save the entire model to a HDF5 file.
# The '.h5' extension indicates that the model shuold be saved to HDF5.
model.save('my_model.h5')

# Recreate the exact same model, including its weights and the optimizer
new_model = tf.keras.models.load_model('my_model.h5')

# Show the model architecture
new_model.summary()

loss, acc = new_model.evaluate(test_images,  test_labels, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100*acc))
```
<br>
```code
Train on 1000 samples
Epoch 1/5
1000/1000 [==============================] - 0s 240us/sample - loss: 1.1880 - accuracy: 0.6560

...

Model: "sequential_9"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_18 (Dense)             (None, 512)               401920    
_________________________________________________________________
dropout_9 (Dropout)          (None, 512)               0         
_________________________________________________________________
dense_19 (Dense)             (None, 10)                5130      
=================================================================
Total params: 407,050
Trainable params: 407,050
Non-trainable params: 0
_________________________________________________________________
1000/1 - 0s - loss: 0.4504 - accuracy: 0.8680
Restored model, accuracy: 86.80%
```
<br>
<br><br>

#### SaveModel Format
위에서는 다음과 같은 내용을 포함한다고 하였습니다.  
- The weight values: .ckpt File 같은 Weights 정보
- The model's configuratrion: Graph의 정보
- The optimizer configuration

위의 3가지의 정보를 하나의 .h5 File이아닌 Directory에 나누어서 담는 방법입니다.  
먼저 결과부터 살펴보면 Model을 저장하는 Directory의 구조는 다음과 같습니다.  

- my_model
 - assets: Model을 돌리는데 필요한 임의의 파일을 저장합니다. Ex) a vocabulary file used initialize a lookup table.
 - variables: 모델의 변수
   - variables.data
   - variables.index
 - .pb: 모델의 변수 + 구조(전체 그래프)

```python
# Create and train a new model instance.
model = create_model()
model.fit(train_images, train_labels, epochs=5)

# Save the entire model as a SavedModel.
!mkdir -p saved_model
model.save('saved_model/my_model') 

# my_model directory
!ls saved_model

# Contains an assets folder, saved_model.pb, and variables folder.
!ls saved_model/my_model

new_model = tf.keras.models.load_model('saved_model/my_model')

# Check its architecture
new_model.summary()

# Evaluate the restored model
loss, acc = new_model.evaluate(test_images,  test_labels, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100*acc))

print(new_model.predict(test_images).shape)
```
<br>
```code
Train on 1000 samples
Epoch 1/5
1000/1000 [==============================] - 0s 238us/sample - loss: 1.1917 - accuracy: 0.6490
Epoch 2/5
1000/1000 [==============================] - 0s 56us/sample - loss: 0.4432 - accuracy: 0.8740
Epoch 3/5
1000/1000 [==============================] - 0s 57us/sample - loss: 0.2925 - accuracy: 0.9250
Epoch 4/5
1000/1000 [==============================] - 0s 52us/sample - loss: 0.2074 - accuracy: 0.9560
Epoch 5/5
1000/1000 [==============================] - 0s 55us/sample - loss: 0.1540 - accuracy: 0.9670
INFO:tensorflow:Assets written to: saved_model/my_model/assets
my_model
assets	saved_model.pb	variables
Model: "sequential_10"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_20 (Dense)             (None, 512)               401920    
_________________________________________________________________
dropout_10 (Dropout)         (None, 512)               0         
_________________________________________________________________
dense_21 (Dense)             (None, 10)                5130      
=================================================================
Total params: 407,050
Trainable params: 407,050
Non-trainable params: 0
_________________________________________________________________
1000/1 - 0s - loss: 0.4745 - accuracy: 0.8730
Restored model, accuracy: 87.30%
(1000, 10))
```
<br>


<hr>
참조: <a href="https://github.com/wjddyd66/Tensorflow2.0/blob/master/SaveAndLoadModels.ipynb">원본코드</a><br>
참조: <a href="https://www.tensorflow.org/tutorials/keras/save_and_load">Save and load models</a><br>
코드에 문제가 있거나 궁금한 점이 있으면 wjddyd66@naver.com으로  Mail을 남겨주세요.

