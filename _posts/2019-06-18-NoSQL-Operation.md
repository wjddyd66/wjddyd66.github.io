---
layout: post
title:  "MongoDB-연산자"
date:   2019-06-18 12:50:00 +0700
categories: [NoSQL]
---
###  비교 연산자
문서를 입력된 값과 비교하여 조회하기 위한 연산자  
<link rel = "stylesheet" href ="/static/css/bootstrap.min.css">
<table class="table">
	<tbody>
	<tr>
		<td>$gt</td><td>기준 값 보다 크다</td>
	</tr>
	<tr>
		<td>$gte</td><td>기준 값 보다 크거나 같다</td>
	</tr>
		<tr>
		<td>$lt</td><td>기준 값 보다 작다</td>
	</tr>
		<tr>
		<td>$lte</td><td>기준 값 보다 작거나 같다</td>
	</tr>
		<tr>
		<td>$ne</td><td>같지 않다</td>
	</tr>
		<tr>
		<td>$ninl</td><td>존재하지 않는다</td>
	</tr>
	</tbody>
</table>
<br>

```js
//비교 연산자 - 문서를 입력된 값과 비교하여 조회하기 위한 연산자
//$gt: 기준 값 보다 크다, $gte: 기준 값 보다 크거나 같다
db.user.find({"age":{$gt:15}})
db.user.find({"age":{$gte:15}})
//$lt: 기준 값 보다 작다, $lte: 기준값 보다 작거나 크다.
db.user.find({"age":{$lt:15}})
db.user.find({"age":{$lte:15}})
//$ne: 같지 않다
db.user.find({"age":{$ne:15}})
```
<br>
결과 - db.user.find({"age":{$lt:15}})
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/NoSQL/C_Operator1.PNG" height="150" width="600" /></div>

<br>
결과 - db.user.find({"age":{$ne:15}})
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/NoSQL/C_Operator2.PNG" height="150" width="600" /></div><br>

###  판단연산자
여러가지 조건을 연결하거나 존재 여부를 판단하는 연산자
<table class="table">
	<tbody>
	<tr>
		<td>$and</td><td>여러 조건을 모두 만족</td>
	</tr>
	<tr>
		<td>$or</td><td>여러 조건 중 하나 이상 만족</td>
	</tr>
		<tr>
		<td>$nor</td><td>Not OR</td>
	</tr>
		<tr>
		<td>$not</td><td>조건을 만족 시키지 않는 것</td>
	</tr>
	</tbody>
</table>
<br>

```js
//판단 연산자 - 여러가지 조건을 연결하거나 존재 여부를 판단하는 연산자 이다
//$and: 여러 조건을 모두 만족
db.user.find({$and :[{"age": {$gt:10}},{"age": {$lt:15}}]})
//$or: 여러 조건 중 하나 이상 만족
db.user.find({$or :[{"age": {$gt:20}},{"age": {$lt:15}}]})
//$nor: Nor or
db.user.find({$nor :[{"age": {$gt:20}},{"age": {$lt:15}}]})
//$not: 조건을 만족 시키지 않는 것
db.user.find({age:{$not:{$lte:15}}})
```
<br>
결과 - db.user.find({$and :[{"age": {$gt:10}},{"age": {$lt:15}}]})
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/NoSQL/J_Operator1.PNG" height="150" width="600" /></div>

<br>
결과 - db.user.find({age:{$not:{$lte:15}}})
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/NoSQL/J_Operator2.PNG" height="150" width="600" /></div><br>

###  SubDocument
Mongo Db는 Collection내의 Document 단위로 검색  
Document안에 Bson 형식인 Document존재시 SubDocument를 활용하여 검색 가능  
<span style ="color: red">$elemMatch를 사용한다.</span>


```js
/*
SubDocument
Mongo Db는 Collection내의 Document 단위로 검색
Document안에 Bson 형식인 Document존재시 SubDocument를 활용하여 검색 가능
$elemMatch를 사용한다.
*/
db.user.insert({"name":"Hwang", "age":18,"Detail":[{"email":"wjddyd66@naver.com","phone":"010-8947-2534"},{"email":"wjddyd66@naver.com2","phone":"010-8947-2534"}]})
db.user.find({"Detail":{$elemMatch:{"email":"wjddyd66@naver.com"}}})
```
<br>
결과:
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/NoSQL/SubDocument.PNG" height="150" width="600" /></div><br>

<hr>
참조: <a href="https://github.com/wjddyd66/NoSQL/tree/master/Operation">원본코드</a><br>
코드에 문제가 있거나 궁금한 점이 있으면 wjddyd66@naver.com으로  Mail을 남겨주세요.