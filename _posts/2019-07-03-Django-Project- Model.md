---
layout: post
title:  "Django-Project-Model 선정"
date:   2019-07-03 09:50:00 +0700
categories: [Project]
---

###  Model 선정
분류를 하기 위하여 6가지의 Classifier Model을 선택하고 각각을 Trainning 및 Parameter를 변화시키며 Accuracy를 확인하였다.  
Train DataSet 과 Test DataSet을 분리하여 Overfitting의 여부를 확인하였다.  
<span style ="color: red">**원래 Code는 Jupyter Notebook으로 한 Shell씩 실행시킨 코드를 GitHub에 올리기 위하여 모아둔 것이다.**</span><br>
```python
#Model.py

#Module Import
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

#DataSet 불러오기
data = pd.read_excel('Z_Data.xlsx')
data2 = pd.read_excel('X_Data.xlsx')

#Train, Test Data로 분리
X_train, X_test, y_train, y_test = train_test_split(data[['People','Univ','Park','Popular','Road','River']], data[['Count']], test_size=0.3, random_state=0)
X_train2, X_test2, y_train2, y_test2 = train_test_split(data2[['People','Park','Popular','Road','River']], data2[['Count']], test_size=0.3, random_state=0)

#GradientBoostingClassifier 결과 확인
from sklearn.ensemble import GradientBoostingClassifier
gbrt = GradientBoostingClassifier(random_state=0)
gbrt.fit(X_train, y_train)

print("훈련 세트 정확도: {:.3f}".format(gbrt.score(X_train, y_train)))
print("테스트 세트 정확도: {:.3f}".format(gbrt.score(X_test, y_test)))

#GradientBoostingClassifier Parameter 변화
max =0
count1 =0
count2 =0
for i in range(1,10):
    for j in np.arange(0.001,0.9,0.001):
        gbrt = GradientBoostingClassifier(random_state=0,max_depth=i,learning_rate=j)
        gbrt.fit(X_train, y_train)
        if(gbrt.score(X_test, y_test)>max):
            max = gbrt.score(X_test, y_test)
            count1 =i
            count2 =j
            print("훈련 세트 정확도: {:.3f}".format(gbrt.score(X_train, y_train)))
            print("테스트 세트 정확도: {:.3f}".format(gbrt.score(X_test, y_test)))
            print('count1',i)
            print('count2',j)


#RandomForestClassifier 결과 확인
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=100, random_state=0)
forest.fit(X_train, y_train)

print("훈련 세트 정확도: {:.3f}".format(forest.score(X_train, y_train)))
print("테스트 세트 정확도: {:.3f}".format(forest.score(X_test, y_test)))

#RandomForestClassifier Parameter 변화
max =0
count1 =0
for i in range(1,400):
    forest = RandomForestClassifier(n_estimators=i, random_state=0)
    forest.fit(X_train, y_train)
    if(forest.score(X_test, y_test)>max):
        max = forest.score(X_test, y_test)
        count1 =i
        print("훈련 세트 정확도: {:.3f}".format(forest.score(X_train, y_train)))
        print("테스트 세트 정확도: {:.3f}".format(forest.score(X_test, y_test)))



#KNeighborsClassifier Parameter 변화 및 결과 확인
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pylab as plt

training_accuracy = []
test_accuracy = []
# 1에서 10까지 n_neighbors를 적용
neighbors_settings = range(1, 30)

for n_neighbors in neighbors_settings:
    # 모델 생성
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train, y_train)
    # 훈련 세트 정확도 저장
    training_accuracy.append(clf.score(X_train, y_train))
    # 일반화 정확도 저장
    test_accuracy.append(clf.score(X_test, y_test))

plt.plot(neighbors_settings, training_accuracy, label="Train Accuarcy")
plt.plot(neighbors_settings, test_accuracy, label="Test Accuarcy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()

#DecisionTreeClassifier 결과 확인
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train, y_train)
print("훈련 세트 정확도: {:.3f}".format(tree.score(X_train, y_train)))
print("테스트 세트 정확도: {:.3f}".format(tree.score(X_test, y_test)))

#DecisionTreeClassifier Parameter 변화
max =0
count1 =0
for i in range(1,10):
    tree = DecisionTreeClassifier(random_state=0,max_depth=i)
    tree.fit(X_train, y_train)
    if(tree.score(X_test, y_test)>max):
        max = tree.score(X_test, y_test)
        count1 =i
        print("훈련 세트 정확도: {:.3f}".format(tree.score(X_train, y_train)))
        print("테스트 세트 정확도: {:.3f}".format(tree.score(X_test, y_test)))

#SVC 결과 확인
from sklearn.svm import SVC
svc = SVC()
svc.fit(X_train2, y_train2)

print("훈련 세트 정확도: {:.2f}".format(svc.score(X_train2, y_train2)))
print("테스트 세트 정확도: {:.2f}".format(svc.score(X_test2, y_test2)))

#SVC Parameter 변화
max =0
count1 =0
count2 =0
for i in np.arange(0.1,10,0.1):
    for j in np.arange(0.1,1000,0.1):
        svc = SVC(C=j,gamma=i)
        svc.fit(X_train2, y_train2)
        if(svc.score(X_test2, y_test2)>max):
            max = svc.score(X_test2, y_test2)
            count1 =i
            count2 =j
            print("훈련 세트 정확도: {:.2f}".format(svc.score(X_train2, y_train2)))
            print("테스트 세트 정확도: {:.2f}".format(svc.score(X_test2, y_test2)))
            print('count1',i)
            print('count2',j)


#MLPClassifier #SVC Parameter 변화 및 결과 확인
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

mlp = MLPClassifier(max_iter=100)

parameter_space = {
    'hidden_layer_sizes': [(30,30,30),(30,10,10),(30,10,20),(30,20,20),(30,5,5),(5,30,30)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive'],
}

clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3)
clf.fit(X_train2, y_train2)

print('Best parameters found:\n', clf.best_params_)

mlp = MLPClassifier(max_iter=3000, alpha= 0.0001, activation= 'tanh', solver= 'adam', learning_rate= 'adaptive', hidden_layer_sizes= (5, 30, 30))
mlp.fit(X_train2,y_train2)
print("훈련 세트 정확도: {:.3f}".format(mlp.score(X_train2, y_train2)))
print("테스트 세트 정확도: {:.3f}".format(mlp.score(X_test2, y_test2))) 
```

결과  
<table class="table" style="width:100%">
	<tbody>
	<tr>
		<td>모델 명</td><td>정확도(Traing)</td><td>정확도(Test)</td><td>과적합 여부</td>
	</tr>
	<tr>
		<td><span style ="color: red">MLPClassifier</span></td>
		<td><span style ="color: red">61.6 %</span></td>
		<td><span style ="color: red">61 %</span></td>
		<td><span style ="color: red">X</span></td>
	</tr>
	<tr>
		<td>GradientBoostingClassifie</td>
		<td>54.6 %</td>
		<td>52.2 %</td>
		<td>X</td>
	</tr>
	<tr>
		<td>K-NN</td>
		<td>57 %</td>
		<td>50 %</td>
		<td>O</td>
	</tr>
	<tr>
		<td>Decision Tree</td>
		<td>54 %</td>
		<td>50 %</td>
		<td>X</td>
	</tr>
	<tr>
		<td>SVM</td>
		<td>53 %</td>
		<td>52 %</td>
		<td>X</td>
	</tr>
	<tr>
		<td>Random Forest</td>
		<td>99.2 %</td>
		<td>52 %</td>
		<td>O</td>
	</tr>
</tbody>
</table>
<br>
<br>
정확도가 가장높고, 과적합 하지 않은 모델 선정<span style ="color: red">**MLPClassifier**</span>가 모델로 선정  
MLPClassifier  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Project/Django9.PNG" height="100%" width="100%" /></div>
1. 모델 구조 및 가정에서 최소의 요구를 가지고 있는 광범위한 예측 모델과 근사
2. 모델  해석 가능성이 낮지만 좋은 <span style ="color: red">**예측력**</span>을 확보할 수 있음  
<hr>
참조:<a href="https://github.com/wjddyd66/Project/tree/master/Django">원본코드</a><br>
코드에 문제가 있거나 궁금한 점이 있으면 wjddyd66@naver.com으로  Mail을 남겨주세요.