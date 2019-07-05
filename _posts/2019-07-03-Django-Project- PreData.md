---
layout: post
title:  "Django-Project-데이터 전처리"
date:   2019-07-03 09:30:00 +0700
categories: [Django,Python]
---

###  데이터 전처리
<span style ="color: red">**Euclidean**</span>거리 계산 방식 이용  
대여소로부터 최단거리의 “대학교, 자전거도로, 관광명소,  공원, 강”까지의 거리를 도출  
<span style ="color: red">**Euclidean**</span>거리 계산 방식 이용  
대여소로부터 1.5km내, 관측소들의 유동인구  평균을 도출  

데이터 전처리
```python
#Pre_Data.py

```
<br>
<br>

빌린 횟수의 경우 <span style ="color: red">**범주형(1~5)사이 값으로 바꿈**</span>  
```R
#Regulation.R
cen<-length(data$count)/5
for(i in 1:length(data2$count)){
  if(i<cen)
    data2$count[i] <-5
  else if(i<cen*2)
    data2$count[i] <-4
  else if(i<cen*3)
    data2$count[i] <-3
  else if(i<cen*4)
    data2$count[i] <-2
  else
  	data2$count[i] <-1
}
```
결과  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Project/Django23.PNG" height="200" width="500" /></div><br>

###  데이터 정규화
알맞은 Model을 선택하기 위하여 Z 정규화와 MIN-MAX Normalization 두개를 사용하여 정규화를 진행하였다.  
정규화 과정  
```R
library(readxl)
library(xlsx)
data = read_excel('Data.xlsx')


#Z변환으로 정규화 후 Data 저장
data$People <- scale(data$People)
data$Univ <- scale(data$Univ)
data$Park <- scale(data$Park)
data$Road <- scale(data$Road)
data$River <- scale(data$River)
data$Popular <- scale(data$Popular)

write.xlsx(data,'Z_Data.xlsx')


#[0-1]변환으로 정규화 후 Data 저장
data$People <- (data$People-min(data$People))/(max(data$People)-min(data$People))
data$Univ <- (data$Univ-min(data$Univ))/(max(data$Univ)-min(data$Univ))
data$Park <- (data$Park-min(data$Park))/(max(data$Park)-min(data$Park))
data$Road <- (data$Road-min(data$Road))/(max(data$Road)-min(data$Road))
data$River <- (data$River-min(data$River))/(max(data$River)-min(data$River))
data$Popular <- (data$Popular-min(data$Popular))/(max(data$Popular)-min(data$Popular))

write.xlsx(data,'X_Data.xlsx')
```
<br>
Z_Data 사진  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Project/Django21.PNG" height="200" width="500" /></div><br>
X_Data 사진  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Project/Django22.PNG" height="200" width="500" /></div><br>

데이터 2차 가공 Data:<https://github.com/wjddyd66/Project/tree/master/Django/Pre_Data>

<hr>
참조:<https://github.com/wjddyd66/Project/tree/master/Django><br>
코드에 문제가 있거나 궁금한 점이 있으면 wjddyd66@naver.com으로  Mail을 남겨주세요.