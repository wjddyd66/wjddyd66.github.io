---
layout: post
title:  "MongoDB-MapReduce"
date:   2019-06-19 08:00:00 +0700
categories: [NoSQL]
---

###  MapReduce
대용량 데이터 베이스를 응축하기 위한 데이터 프로세싱 패러다임  
Ex) 게시판에서 사용자들이 이용했던 로그를 이용해 누가 활동적인지 파악 가능  
<div><img src="https://t1.daumcdn.net/cfile/tistory/27571135574E6E4630" height="300" width="600" /></div><br>
출처:<a href="https://cocomo.tistory.com/361?category=686190">Cocomo Coding 블로그</a> <br>

<link rel = "stylesheet" href ="/static/css/bootstrap.min.css">
<table class="table">
	<tbody>
	<tr>
		<td>Map</td><td>Map은 단순한 Key,Value형식으로 선언하며, 이 값들은 emit을 통해 Reduce로 전달</td>
	</tr>
	<tr>
		<td>Reduce</td><td>데이터를 감소 시키는 역할을 한다. Reduce를 통해 특정 값을 뽑아 낸다.</td>
	</tr>
		<tr>
		<td>Query</td><td>Map에 들어갈 Collection을 필터링 할 때 사용</td>
	</tr>
		<tr>
		<td>Out</td><td>결과를 담을 컬랙션 명</td>
	</tr>
	</tbody>
</table>
<br>

```js
//MapReduce: 대용량 데이터 베이스를 응축하기 위한 데이터 프로세싱 패러다임
//자료 준비
use mr
db.bank.insert({
    "cust_id" : "A123",
    "amount" : 500,
    "status" : "A"
})
 
db.bank.insert({
    "cust_id" : "A123",
    "amount" : 2500,
    "status" : "A"
})
 
db.bank.insert({
    "cust_id" : "B212",
    "amount" : 200,
    "status" : "A"
})
 
db.bank.insert({
    "cust_id" : "A123",
    "amount" : 300,
    "status" : "D"
})

//자료 확인:
db.bank.find()

//MapReduce
//make map
db.bank.mapReduce(
    //map은 단순한 key value 형식으로 선언하며, 이 값들을 emit을 통해 reduce로 전달한다.
    function() { emit ( this.cust_id, this.amount ) },
    //Reduce은 데이터를 감소 시키는 역활을 한다.
    //Reduce를 통해 특정 값을 뽑아 낸다.
    function( key, values ) { return Array.sum( values ) },
    {
        //query: Map에 들어갈 Collection을 필터링 할때 사용
        query : {status : "A"},
        //결과를 담을 컬랙션 명
        out : "order_totals"
    }
)
```
<br>
결과 - MapReduce 처리 전 Collection
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/NoSQL/MapReduce1.PNG" height="150" width="600" /></div>
<br>
결과 - MapReduce 처리 후 결과
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/NoSQL/MapReduce2.PNG" height="150" width="600" /></div>
<br>

<hr>
내용 참조: <a href="https://cocomo.tistory.com/361?category=686190">Cocomo Coing 블로그</a><br>
참조: <a href="https://github.com/wjddyd66/NoSQL/tree/master/MapReduce">원본코드</a><br>
코드에 문제가 있거나 궁금한 점이 있으면 wjddyd66@naver.com으로  Mail을 남겨주세요.