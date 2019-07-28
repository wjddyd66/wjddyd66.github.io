---
layout: post
title:  "R-교차검증"
date:   2019-06-17 12:40:20 +0700
categories: [R]
---

###  교차검증(K-Fold Cross Validation)
K 개의 fold를 만들어서 진행하는 교차 검증  
사용이유  
<ul>
	<li>총 데이터 갯수가 적은 데이터 셋에 대하여 정확도를 향상시킬 수 있음</li>
	<li>기존에 Training, Validation, Test 세 개의 집단으로 분류하는 것보다, Trainning 과 Test로만 분류할 때 학습 데이터 셋이 더 많기 때문이다.</li>
	<li>데이터 수가 적은데 검증과 테스트에 데이터를 더 뺏기면 underfitting 등 성능이 미달되는 모델이 학습됨</li>
</ul>  
<div><img src="https://www.researchgate.net/profile/B_Aksasse/publication/326866871/figure/fig2/AS:669601385947145@1536656819574/K-fold-cross-validation-In-addition-we-outline-an-overview-of-the-different-metrics-used.jpg" height="300" width="600" /></div><br>

참조: <a href="https://www.researchgate.net/figure/K-fold-cross-validation-In-addition-we-outline-an-overview-of-the-different-metrics-used_fig2_326866871">ResearchGate</a>  

과정  

<ul>
	<li>기존 과정과 같이 Trainning Set과 Test Set을 나눈다.</li>
	<li>Trainning 을 K 개의 fold로 나눈다.</li>
	<li>한 개의 Fold에 있는 데이터를 다시 K 개로 쪼갠다음, K-1개는 Trainning Data, 마지막 한개는 Validation Data Set으로 지정한다.</li>
	<li>모델을 생성하고 예측을 진행하여, 이에 대한 에러값을 추출한다.</li>
	<li>다음 Fold에서는 Validation 셋을 바꿔서 지정하고, 이전 Fold에서 Validation 역할을 했던 Set은 다시 Trainning set으로 활용한다.</li>
	<li>이를 K번 반복한다.</li>
</ul> 
참조: <a href="https://nonmeyet.tistory.com/entry/KFold-Cross-Validation%EA%B5%90%EC%B0%A8%EA%B2%80%EC%A6%9D-%EC%A0%95%EC%9D%98-%EB%B0%8F-%EC%84%A4%EB%AA%85">nomeyet 블로그</a>

```R
#K Fold Cross Validation
#패키지 설치 및 불러오기
install.packages("cvTools")
library(cvTools)

#K Fold Data 만들기
set.seed(12)
cross<-cvFolds(n=6,K=3,R=1,type="random")
cross

#Data 확인 및 가공
str(cross)
names(cross)

cross$subsets
cross$which

cross$subsets[cross$which==1,1]
cross$subsets[cross$which==2,1]
cross$subsets[cross$which==3,1]
cross

#K Fold Cross Validation Setting
set.seed(123)
cross<-cvFolds(n=nrow(iris),K=3,R=1,type="random")
cross

acc<-numeric()
cnt<-1
r=1
k=1:3

#K Fold Cross Validation 수행
for(i in k){
  idx<-cross$subsets[cross$which==i,r]
  #cat("test:",i,"검정데이터\n")
  #print(iris[idx,])
  test<-iris[idx,]
  for(j in k[-1]){
    idx<-cross$subsets[cross$which==j,r]
    #cat("test:",i,"훈련데이터\n")
    train<-iris[idx,]
    model<-naiveBayes(Species~.,data=train)
    pred<-predict(model,test)
    t<-table(pred,test$Species)
    acc[cnt]<-(t[1,1]+t[2,2]+t[3,3])/sum(t)
    cnt<-cnt+1
  }
}
acc
mean(acc)

```

결과:
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/R/K_Fold.PNG" height="150" width="600" /></div><br>
<hr>
참조: <a href="https://github.com/wjddyd66/R/tree/master/K-Fold-Cross-Validation">원본코드</a><br>
코드에 문제가 있거나 궁금한 점이 있으면 wjddyd66@naver.com으로  Mail을 남겨주세요.