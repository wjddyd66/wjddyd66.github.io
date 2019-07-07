---
layout: post
title:  "Django-Project-Model Parameter"
date:   2019-07-03 10:00:00 +0700
categories: [Project]
---

###  Model Parameter
Model 분류값은 1~5의 5개의 Class로 분류 할 경우  
2~5의 값을 가지는 Class를 판별할 때 많은 오차가 발생되는 것을 발견  
<span style ="color: red">**상위  20%  = “상”**</span>, <span style ="color: blue">**하위  10%  = “하”**</span>나머지는 ="중"으로서 3개의 Class로서 분류로 바꾸었다.  
 - ‘상’   등급  :  지정  위치에  대여소  설치  추천
 - ‘중’   등급:  내부   검토에  의해   설치
 - ‘하’   등급:   설치  지양

<span style ="color: red">**GridSearchCV**</span>를 활용하여 사용할 MLPClassifier의 최적은 Parameter를 찾는 코드이다.  
GridSearchCV: 클래스 객체에 fit 메서드를 호출하면 grid search를 사용하여 자동으로 복수개의 내부 모형을 생성하고 이를 모두 실행시켜서 최적 파라미터를 찾아준다. 생성된 복수개와 내부 모형과 실행 결과는 다음 속성에 저장된다.  

<span style ="color: red">**원래 Code는 Jupyter Notebook으로 한 Shell씩 실행시킨 코드를 GitHub에 올리기 위하여 모아둔 것이다.**</span><br>

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV 

#Data
data = pd.read_excel('Final_Data.xlsx')
X_train, X_test, y_train, y_test = train_test_split(data[['People','Park','Popular','Road','River','Univ']], data[['Count']], test_size=0.3, random_state=50) 

 

#Parameter Select
mlp = MLPClassifier(max_iter=100)

parameter_space = {
    'hidden_layer_sizes': [(20, 10, 20),(10,10,20),(10,10,30),(20,5,5)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive'],
}

clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3)
clf.fit(X_train, y_train)

#Parameter Check
print('Best parameters found:\n', clf.best_params_)

#Model Accuracy Test
mlp = MLPClassifier(max_iter=3000, alpha= 0.0001, activation= 'relu', solver= 'adam', learning_rate= 'adaptive', hidden_layer_sizes= (10, 10, 30))
mlp.fit(X_train,y_train)

#X_train2, X_test2, y_train2, y_test2
print("훈련 세트 정확도: {:.3f}".format(mlp.score(X_train, y_train)))
print("테스트 세트 정확도: {:.3f}".format(mlp.score(X_test, y_test)))

#Model 저장
from sklearn.externals import joblib
file_name = 'model.pkl' 
joblib.dump(mlp, file_name) 
```
<br>
<span style ="color: red">**Trainning 된 Model을 가져다가 사용하기 위하여 .pkl파일로 저장**</span><br>
.pkl파일: 파이썬 객체를 그대로 파일에 저장하고 다시 파일에서 읽어들이기 위한 확장자

결과  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Project/Django24.PNG" height="250" width="400" /></div>

###  Model Parameter 후보 및 최종 선택 결과

<table class="table" style="width:100%">
	<tbody>
	<tr>
		<td>Parameter</td><td>설명</td><td>후보</td><td>최종값</td>
	</tr>
	<tr>
		<td>Hidden_Layer_Sizer</td>
		<td>Hidden Layer 크기 설정</td>
		<td>계속하여 변경</td>
		<td><span style ="color: red">(10,10,30)</span></td>
	</tr>
	<tr>
		<td>max_iter</td>
		<td>최대 반복 횟수</td>
		<td>1500~3000</td>
		<td><span style ="color: red">3000</span></td>
	</tr>
	<tr>
		<td>alpha</td>
		<td>L2 Regulation penalty</td>
		<td>0.0001,0.05</td>
		<td><span style ="color: red">0.0001</span></td>
	</tr>
	<tr>
		<td>activation</td>
		<td>활성 함수</td>
		<td>tanh, relu</td>
		<td><span style ="color: red">relu</span></td>
	</tr>
	<tr>
		<td>solver</td>
		<td>weight optimizer</td>
		<td>adam, sgd</td>
		<td><span style ="color: red">adam</span></td>
	</tr>
	<tr>
		<td>learning_rate</td>
		<td>Schedule for weight updates</td>
		<td>adaptive, constant</td>
		<td><span style ="color: red">adaptive</span></td>
	</tr>
</tbody>
</table>
<br>


<br>

<hr>
참조:<a href="https://github.com/wjddyd66/Project/tree/master/Django">원본코드</a><br>
코드에 문제가 있거나 궁금한 점이 있으면 wjddyd66@naver.com으로  Mail을 남겨주세요.