---
layout: post
title:  "R-카이제곱 검정"
date:   2019-06-17 08:40:20 +0700
categories: [R]
---

###  카이제곱 검정방법
카이제곱 데스트는 그룹간에 차이가 있는지 여부(= 그룹끼지 독립이 아닌지의 여부)에 대해 Chisquare 분포를 사용해 가설검정을 하는 방법이다. 그룹간에 차이가 있는지 없는지의 여부라는 의미는 그룹간의 비율차이가 있는지의 여부라는 의미이다.  

<span style ="color: red">**독립변수: 범주형, 종속변수: 범주형**</span><br>

카이제곱의 검정 방법은 목적에 따라서 3가지로 크게 나눌수 있다.  

1. 독립성 검정: 두 변수는 서로 연관성이 있는가 없는가?
2. 적합성 검정: 실제 표본이 내가 생각하는 분포와 같은가 다른가?
3. 동일성 검정: 두 집단의 분포가 동일한가? 다른 분포인가? 


참조: <a href="https://m.blog.naver.com/PostView.nhn?blogId=leerider&logNo=100189714605&proxyReferer=https%3A%2F%2Fwww.google.co.kr%2F">leerider 블로그</a>  

###  카이제곱 기본 이해
<link rel = "stylesheet" href ="/static/css/bootstrap.min.css">

<table class="table">
	<tbody>
	<tr>
		<td>귀무가설</td><td>공부와 합격은 상관이 없다.</td>
	</tr>

	<tr>
		<td>대립가설</td><td>공부와 합격은 상관이 있다.</td>
	</tr>
	
	<tr>
		<td>결과</td><td>p-value(0.083) > 0.05(95% 신뢰확률에서의 유의수준) => 귀무가설 채택  </td>
	</tr>
	</tbody>
</table>

<br>
```R
#카이제곱분석(교차분석) 기본이해

#카이제곱을 위한 패키지 설치
install.packages("gmodels")
library(gmodels)
#데이터 불러오기
study<-read.csv("C:/git/R/Data/pass_cross.csv")
head(study)
#데이터 가공 및 카이제곱 결과 확인
table(study$공부함,study$합격)
table(study$공부안함,study$불합격)
CrossTable(study$공부함,study$합격,chisq=T)
```
<br>

출력결과

<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/R/Chisquare1.PNG" height="150" width="600" /></div><br>

###  카이제곱 기본 이해
<table class="table">
	<tbody>
	<tr>
		<td>귀무가설</td><td>현재 사용중인 주사위는 게임에 적합하다.(1~6까지의 나오는 확률이 비슷하다.)</td>
	</tr>

	<tr>
		<td>대립가설</td><td>현재 사용중인 주사위는 게임에 적합하지 않다.(1~6까지의 나오는 확률이 다르다..)</td>
	</tr>
	
	<tr>
		<td>결과</td><td>p-value(0.01439)
	    <
	    0.05(95% 신뢰확률에서의 유의수준)  =>귀무가설 채택  </td>
	</tr>
	</tbody>
</table>

<br>
```R
#주사이 카지에곱 검정
#내가 생각하는 주사위는 나올확률이 동일하다.
chisq.test(c(4,6,17,16,8,9))
```
<br>

출력결과

<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/R/Chisquare2.PNG" height="150" width="600" /></div><br>

###  카이제곱 독립성 검정
<table class="table">
	<tbody>
	<tr>
		<td>귀무가설</td><td>부모의 학력수준과 자녀의 대학진학여부 간의 관련이 없다.</td>
	</tr>

	<tr>
		<td>대립가설</td><td>부모의 학력수준과 자녀의 대학진학여부 간의 관련이 있다.</td>
	</tr>
	
	<tr>
		<td>결과</td><td>p-value(0.2507057) > 0.05(95% 신뢰확률에서의 유의수준) => 귀무가설 채택</td>
	</tr>
	</tbody>
</table>

<br>
```R
#독립성(관련성): 두 속성 간의 관계검정
data<-read.csv("C:/git/R/Data/cleanDescriptive.csv",header = T,fileEncoding = "UTF-8")
head(data)

#부모의 학력수준과 자녀의 대학여부 간 관련성 검정
x<-data$level2 #부모의 학력수준(독립변수:영향줌)
y<-data$pass2 #자녀의 대학진학여부(종속변수:영향받음)

result<-data.frame(level=x,pass=y)
dim(result)
table(result)

chisq.test(x,y,correct = F) #correct = F : 연속성 보정 미적용(기본값:T-연속성보정 적용)
```
<br>

출력결과

<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/R/Chisquare3.PNG" height="150" width="600" /></div><br>

###  카이제곱 동질성 검정
<table class="table">
	<tbody>
	<tr>
		<td>귀무가설</td><td>교육방법에 따른 교육생들의 만족도 차이가 없다.</td>
	</tr>

	<tr>
		<td>대립가설</td><td>교육방법에 따른 교육생들의 만족도 차이가 있다.</td>
	</tr>
	
	<tr>
		<td>결과</td><td>p-value(0.5865) > 0.05(95% 신뢰확률에서의 유의수준) => 귀무가설 채택</td>
	</tr>
	</tbody>
</table>

<br>
```R
#부모의 학력수준과 자녀의 대학여부 간 관련성 검정
x<-data$level2 #부모의 학력수준(독립변수:영향줌)
y<-data$pass2 #자녀의 대학진학여부(종속변수:영향받음)

result<-data.frame(level=x,pass=y)
dim(result)
table(result)

chisq.test(x,y,correct = F) #correct = F : 연속성 보정 미적용(기본값:T-연속성보정 적용)


#동질성 검정 : 집단 간 분포 동일여부 검정
rm(list=ls())
gc()
data<-read.csv("C:/git/R/Data/homogenity.csv",header = T)
head(data)

#교육방법에 따른 교육생들의 만족도 차이가 있는지 검정
#귀무가설: 교육방법에 따른 교육생들의 만족도 차이가 없다.
#대립가설: 교육방법에 따른 교육생들의 만족도 차이가 있다.
str(data)

data<-subset(data,!is.na(survey),c(method,survey))
data
table(data$method)
table(data$survey)

data$method2[data$method==1]<-"방법1"
data$method2[data$method==2]<-"방법2"
data$method2[data$method==3]<-"방법3"
head(data)

data$survey2[data$survey==1]<-"매우만족"
data$survey2[data$survey==2]<-"만족"
data$survey2[data$survey==3]<-"보통족"
data$survey2[data$survey==4]<-"불만족"
data$survey2[data$survey==5]<-"매우불만족"
head(data)

table(data$method2,data$survey2) #각 집단별 길이가 같아야 한다.
chisq.test(data$method2,data$survey2) 
#X-squared = 6.5447, df = 8, p-value = 0.5865
#해석: p-value(0.5865)>0.05 -> 귀무가설 채택
#결론: 교육방법에 따른 교육생들의 만족도 차이가 없다.

CrossTable(data$method2,data$survey2,chisq = T)
```
<br>

출력결과

<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/R/Chisquare4.PNG" height="150" width="600" /></div><br>

<hr>
참조: <a href="https://github.com/wjddyd66/R/tree/master/Chisquare">원본코드</a><br>
코드에 문제가 있거나 궁금한 점이 있으면 wjddyd66@naver.com으로  Mail을 남겨주세요.

