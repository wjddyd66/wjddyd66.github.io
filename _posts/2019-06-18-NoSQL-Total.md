---
layout: post
title:  "MongoDB-집계"
date:   2019-06-18 12:55:00 +0700
categories: [NoSQL]
---

###  Count
컬랙션 내의 문서 수를 반환하는 집계 도구
1. 전체 문서 count: 조건X
 - 컬랙션의 크기와 상관없이 빠르다.
2. 조건을 만족하는 문서 count: 조건O
 - 쿼리 조건을 추가하면 count는 느려진다.


```js
/*
count: 컬렉션 내의 문서 수를 반환하는 집계 도구
1. 전체 문서 count: 조건X
컬랙션 내 전체 문서의 수를 세는 것은 컬렉션의 크기와 상관없이 빠르다.
2. 조건을 만족하는 문서 count: 조건O
쿼리 조건을 추가하면 count는 느려진다.
*/
//전체 문서 count
db.user.count()
//조건을 만족하는 문서 count
db.user.count({"age":10})
```
<br>
결과 - db.user.count({"age":10})
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/NoSQL/Count.PNG" height="150" width="600" /></div>
<br>

###  Distinct
주어진 키의 고유한 값을 찾는다.  
```js
//distinct: 주어진 키의 고유한 값을 찾는다.
//방법1
db.runCommand({"distinct":"user","key":"name"})
//방법2
db.user.distinct("name")
```
<br>
결과 - db.user.distinct("name")
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/NoSQL/Distinct1.PNG" height="150" width="600" /></div>

###  Group
문서들을 선택한 기준으로 그룹으로 묶어 집계를 낼 때 사용  
<link rel = "stylesheet" href ="/static/css/bootstrap.min.css">
<table class="table">
	<tbody>
	<tr>
		<td>key</td><td>지정한 컬렉션에서 문서를 그룹핑할 키를 지정. Group by column 과 동일함.</td>
	</tr>
	<tr>
		<td>$reduce</td><td>컬렉션 내의 각 문서들에 대해서 한번씩 호출한다. Function의 인자는 총 2개인데,하나는 해당 문서를 받고, 또 다른 하나는 현재까지 누적 계산된 문서를 받는다. 누적계산은 사용자가 정의할 수 있다.</td>
	</tr>
		<tr>
		<td>finalize</td><td>SQL의 Having과 같음. 그룹핑 될 결과로 다시 한번 연산할 때 사용한다.</td>
	</tr>
		<tr>
		<td>initial</td><td>적 계산할 항목을 정의한다.</td>
	</tr>
		<tr>
		<td>condition</td><td>조건을 줄 때 사용한다.</td>
	</tr>
	</tbody>
</table>
<br>
```js
/*
group: 문서들을 선택한 키를 기준으로 그룹으로 묶어 집계를낼 때 사용
key → 지정한 컬렉션에서 문서를 그룹핑할 키를 지정. Group by column 과 동일함.
reduce → 컬렉션 내의 각 문서들에 대해서 한번씩 호출한다. Function의 인자는 총 2개인데,
하나는 해당 문서를 받고, 또 다른 하나는 현재까지 누적 계산된 문서를 받는다. 누적계산은 사용자가 정의할 수 있다.
finalize → SQL의 Having과 같음. 그룹핑 될 결과로 다시 한번 연산할 때 사용한다.
initial → 누적 계산할 항목을 정의한다.
condition → 조건을 줄 때 사용
*/

//group Data 준비 - 10000개 Data 적용
var user2 ={};
function adduser2(){
    for(var i=0;i<100;i++){
    	for(var j=0;j<100;j++){
    	user2.height = i;
    	user2.weight =i+j;
    	db.user2.insert(user2);
    	}
    }
}
adduser2();
db.user2.count();

//Height 90 이상 중 Weight 평균 구하는 쿼리
db.user2.group({
//user2 Colection 에서의 Key
"key":{"height":1},
//결과 result의 초기값 설정
"initial":{"sum":0,"count":0,"avg":0},
//컬랙션 내의 문서를 한번 씩 호출하면서 명령 실행
//result.sum = 키 별 몸무게 합
//retusl.count = Data의 수
"$reduce":function(curr,result){
		if(curr.number == result.number){
			result.sum += curr.weight;
			result.count++;
		}
	},
	//조건: height가 90이상만 실행
	"condition":{"height":{"$gte":90}},
	//SQL의 Having on 역활
	//Group으로 묶은 결과에 함수 적용
	"finalize": function(result){
		//result.avg에 키별 몸무게 평균 적용
		result.avg = result.sum/result.count;
		//필요없는 결과 tag 삭제
		delete result.sum;
		delete result.count;
	}
})
```
<br>
결과:
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/NoSQL/Group.PNG" height="150" width="600" /></div><br>

<hr>
내용 참조: Namoosori-MongoDB(ver2.21) PDF<br>
내용 참조: <a href="https://cocomo.tistory.com/360">Cocomo Coding 블로그</a><br>
참조: <a href="https://github.com/wjddyd66/NoSQL/tree/master/Total">원본코드</a><br>
코드에 문제가 있거나 궁금한 점이 있으면 wjddyd66@naver.com으로  Mail을 남겨주세요.