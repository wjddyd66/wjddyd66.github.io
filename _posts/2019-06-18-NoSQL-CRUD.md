---
layout: post
title:  "MongoDB-CRUD"
date:   2019-06-18 12:40:00 +0700
categories: [NoSQL]
---

###  생성하기(Insert)
새로운 문서를 생성합니다.
1. insert: 단일 또는 다수의 문서를 입력할 때 사용
2. insertOne: 단일 문서를 입력할 때 사용
3. insertMany: 다수의 문서를 입력할 때 사용

```js
/*
생성하기(Insert)
1. insert: 단일 또는 다수의 문서를 입력할 때 사용
2. insertOne: 단일 문서를 입력할 때 사용
3. insertMany: 다수의 문서를 입력할 때 사용
*/
//Document 생성
Hwang = {"name":"JeongYong","description":"Programmer","openDate":new Date()}
//Programmer Collection 생성 후
//Hwang 문서 Programmer Collection에 추가
db.programmer.insert(Hwang)
```
<br>

###  조회하기(find)
기존 문서를 조회합니다.
1. find: 모든 문서 조회
2. findOne: 하나의 문서 조회
3. find(query): query조건에 맞는 문서 조회

```js
/*
조회하기(find)
1. find(): 모든 문서 조회
2. findOne(): 하나의 문서 조회
3. find(query): query조건에 맞는 문서 조회
*/
db.programmer.find()
query = {"name": "JeongYong"}
db.programmer.find(query)
```
<br>
결과 - Insert한 Data 조회
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/NoSQL/Insert.PNG" height="150" width="600" /></div><br>

###  갱신하기(Update)
기존 문서를 수정합니다.
 - 기본적인 방법: Method의 첫번째 인자는 갱신하려는 문서를 찾고, 두번째 인자는 치환할 새로운 문서의 내용이다.


```js
/*
갱신하기(Update)
update는 기존의 문서를 수정하는 것
Method의 첫번째 인자는 갱신하려는 문서를 찾고, 두번째는 치환할 새로운 문서이다.
*/
//갱신하려는 문서의 조건
query1 = {"name": "JeongYong"}
//갱신하려는 문서내용
query2 = {"name": "JeongYong2"}
//update 진행
db.programmer.update(query1,query2)
//결과 확인
db.programmer.find()
```
<br>
결과:
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/NoSQL/Update.PNG" height="150" width="600" /></div><br>

 - 조건을 걸어서 Update하기
<link rel = "stylesheet" href ="/static/css/bootstrap.min.css">
<table class="table">
	<tbody>
	<tr>
		<td>$set</td><td>해당 키의 값만 갱신</td>
	</tr>
	<tr>
		<td>$inc</td><td>해당 키의 값을 증가 또는 감소</td>
	</tr>
		<tr>
		<td>$push</td><td>지정된 키가 존재하면 배열의 끝에 추가하고, 없으면 새로운 배열을 만들어 추가</td>
	</tr>
		<tr>
		<td>$addToSet</td><td>배열 내에 같은 값이 존재하지 않는 경우에만 추가</td>
	</tr>
		<tr>
		<td>$pop</td><td>배열의 요소를 제거 {key:1}인 경우는 배열의 끝에서, {key:-1}인 경우 배열의 처음 요소를 제거</td>
	</tr>
		<tr>
		<td>$pull</td><td>주어진 조건에 일치하는 배열의 요소를 제거</td>
	</tr>
	</tbody>
</table>

<br>

```js
//조건을 걸어서 Update하기
db.programmer.insert(Hwang)
query3 = {"name": "JeongYong2"}
query4 = {$set: {"name": "JeongYong3"}}
db.programmer.update(query3,query4)
db.programmer.find()
```
<br>
결과:
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/NoSQL/Update2.PNG" height="150" width="600" /></div><br>

 - 조건에 맞지 않는 문서가 발견되면 쿼리 문서와 갱신문서를 합친 새로운 문서 입력, 3번째 인자를 True로 한다.


```js
//조건에 맞지 않는 문서가 발견되면 쿼리 문서와 갱신문서를 합친 새로운 문서 입력
//3번째 인자를True로 한다.
query5 = {"name": "JeongYong4"}
query6 = {$set: {"name": "JeongYong5"}}
db.programmer.update(query5,query6,true)
db.programmer.find()
```
<br>
결과:
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/NoSQL/Update3.PNG" height="150" width="600" /></div><br>

 - 다중 문서 갱신, 4번째 인자를 True로 한다.


```js
//다중 문서 갱신 : 4번째 인자를 true로 세팅한다.
db.programmer.insert({"name":"JeongYong3"})
db.programmer.insert({"name":"JeongYong3"})
db.programmer.insert({"name":"JeongYong3"})
query7 = {"name": "JeongYong3"}
query8 = {$set: {"name": "JeongYong6"}}
db.programmer.update(query7,query8,false,true)
db.programmer.find()
```
<br>
결과:
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/NoSQL/Update4.PNG" height="150" width="600" /></div><br>

###  삭제하기(remove)
1. 부분 삭제하기: 조건 O
2. 모두 삭제하기: 조건 X

```js
/*
삭제하기(remove)
1. 부분삭제하기: 조건O
2. 모두 삭제하기: 조건X
*/
//부분 삭제하기
query9 = {"name": "JeongYong5"}
db.programmer.remove(query9)
db.programmer.find()
//모두 삭제하기
db.programmer.remove({})
db.programmer.find()
```
<br>
결과 - 부분 삭제하기:
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/NoSQL/Delete1.PNG" height="150" width="600" /></div><br>
결과 - 모두 삭제하기:
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/NoSQL/Delete2.PNG" height="150" width="600" /></div><br>
<span style ="color: red">**_id: Mongo에서 RDBMS의 PK와 같은 역할을 한다. Document생성시 자동으로 배정받고 Document끼리 중복되지 않는 값 이다.**</span>
<hr>
참조: <a href="https://github.com/wjddyd66/NoSQL/tree/master/CRUD">원본코드</a><br>
코드에 문제가 있거나 궁금한 점이 있으면 wjddyd66@naver.com으로  Mail을 남겨주세요.