---
layout: post
title:  "Django-Project-데이터 변수 설정"
date:   2019-07-03 09:00:00 +0700
categories: [Project]
---

###  따릉이 상위 대여소 파악
따릉이 대여소의 사용량에 따라 상위 대여소와 하위 대여소의 특성을 알아보기 위하여 지도에 찍는 과정이다.  
서울 열린 데이터 광장에서 따릉이 대여소의 위치와 사용량의 데이터를 가지고와서 합치는 작업을 하였다.  
<a href="https://data.seoul.go.kr/search/newSearch.jsp?query=%EA%B3%B5%EA%B3%B5%EC%9E%90%EC%A0%84%EA%B1%B0">데이터 출처</a><br>
서울시 따릉이 대여소 사용량 Data  
<img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Project/Django12.PNG" height="200" width="400" /><br>
서울시 따릉이 대여소 위치 Data  
<img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Project/Django13.PNG" height="200" width="400" /><br>
서울시 따릉이 대여소 사용량과 위치 Data  
<img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Project/Django14.PNG" height="200" width="400" /><br>
<br><br>
위의 데이터를 활용하여 상위100개와 하위 100개의 대여소를 Map에 그려 확인하였다. 

<br>

따릉이 데이터 전처리 과정

<br>

```R
library(ggplot2)
library(ggmap)
library(xlsx)
library(readxl)
library(dplyr)

#데이터 불러오기
location <- read_excel("location.xlsx",na="NA")
count <- read_excel("count.xlsx",na="NA")
#Column 명 변경
colnames(location) <- c('number','name','x','y')
#Column 숫자로 변경
location$number <- as.numeric(location$number)
location$x <- as.numeric(location$x)
location$y <- as.numeric(location$y)

#Count Data에 x,y좌표 추가
for(i in 1:length(count$number)){
  for(j in 1:length(location$x)){
    if(count$number[i] == location$number[j]){
      count$x[i] <- location$x[j]
      count$y[i] <- location$y[j]
    }
  }
}

#Map을 위한 Data 저장 
write.xlsx(people,file="map.xlsx")

#MapData 불러오기
data<-read_excel("map.xlsx",na="NA")

#MapData 상위 100개, 하위100개 나누기
many <- data[1:100,]
min <- data[length(data$number)-936:length(data$number),]

#GoogleMap Key 등록
register_google(key='개인API')

#GoogleMap 을 활용한 Map그리기
center <- c(mean(data$x),mean(data$y))
seoul <- get_map(center, zoom=11, maptype='roadmap')

#상위 100개 지도에 그리기
ggmap(seoul) + geom_point(data=many,aes(x=x,y=y),size=2.5,alpha=0.8,col='red')

#하위 100개 지도에 그리기
ggmap(seoul) + geom_point(data=min,aes(x=x,y=y),size=2.5,alpha=0.8,col='blue')

```
<br>
확인결과  
<img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Project/Django5.PNG" height="300" width="600" /><br>
공공자전거 이용자수 상위 100개소를 지도에 표시한 결과 공원, 강 주변에 위치  
<span style ="color: red">**공원, 강**</span>의 위치를 변수로 설정  

<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Project/Django4.PNG" height="100%" width="100%" /></div>
서울 시설공단 공공자전거운영처의 대여소 설치기준을 참고하여 <span style ="color: red">**유동인구, 대학교, 관광명소, 자전거도로**</span>를 변수로 설정  

###  데이터 출처 및 1차 가공
<link rel = "stylesheet" href ="/static/css/bootstrap.min.css">
<table class="table">
	<tbody>
	<tr>
		<td>데이터</td><td>출처</td><td>1차 가공</td>
	</tr>
	<tr>
		<td>공원</td>
		<td><a href="https://www.tripadvisor.co.kr">tripadvisor</a></td>
		<td>
		<ul>
			<li>상위 12개 선택</li>
			<li>Google Map에 Mapping</li>
			<li>위도, 경도 값 저장</li>
		</ul>
		</td>
	</tr>
	<tr>
		<td>강</td>
		<td>구글 검색</td>
		<td>
		<ul>
			<li>검색 많이된 상위 9개 선택</li>
			<li>특정 길이에 따라 자른뒤 Google Map에 Mapping</li>
			<li>위도, 경도 값 저장</li>
		</ul>
		</td>
	</tr>
	<tr>
		<td>유동인구</td>
		<td><a href="https://data.seoul.go.kr">서울 열린데이터 광장</a></td>
		<td>
        <ul>
			<li>Google Map활용하여 길에대한 위도, 경도 값 찾음</li>
			<li>이름이 같은 길 경우 지도에 그려본 뒤 제거</li>
		</ul>
		</td>
	</tr>
	<tr>
		<td>관광명소</td>
		<td><a href="https://www.tripadvisor.co.kr">tripadvisor</a></td>
		<td>
		<ul>
			<li>상위 20개 선택</li>
			<li>Google Map에 Mapping</li>
			<li>위도, 경도 값 저장</li>
		</ul>
		</td>
	</tr>
	<tr>
		<td>대학교</td>
		<td>전국 대학교 크기 순위</td>
		<td>
		<ul>
			<li>상위 20개 선택</li>
			<li>상관관계 분석 후 낮은값 제거</li>
			<li>Google Map에 Mapping</li>
			<li>위도, 경도 값 저장</li>
		</ul>
		</td>
	</tr>
	<tr>
		<td>자전거도로</td>
		<td>서울시 추천 자전거 도로 검색</td>
		<td>
		<ul>
			<li>상위 30개 선택</li>
			<li>Google Map에 Mapping</li>
			<li>위도, 경도 값 저장</li>
		</ul>
		</td>
	</tr>
</tbody>
</table>

<br>

###  데이터 1차가공 형태
공원  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Project/Django15.PNG" height="200" width="400" /></div>

강  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Project/Django16.PNG" height="200" width="400" /></div>

유동인구  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Project/Django17.PNG" height="200" width="400" /></div>

관광명소  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Project/Django18.PNG" height="200" width="400" /></div>

대학교  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Project/Django19.PNG" height="200" width="400" /></div>

자전거도로  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Project/Django20.PNG" height="200" width="400" /></div>

<br>

<a href="https://github.com/wjddyd66/Project/tree/master/Django/Data">데이터 1차 가공 Data</a>

<hr>
참조:<a href="https://github.com/wjddyd66/Project/commit/a90f557c4e952909c566f45851ab516e24f9a727">원본코드</a><br>
코드에 문제가 있거나 궁금한 점이 있으면 wjddyd66@naver.com으로  Mail을 남겨주세요.