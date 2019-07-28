---
layout: post
title:  "MongoDB-Index"
date:   2019-06-19 08:40:00 +0700
categories: [NoSQL]
---

###  Index
Index는 특정 문서를 탐색할 때 전부를 탐색하지 않고도 데이터를 찾을 수 있게 합니다.(속도 향상)  
특정 데이터를 쉽게 추출할 수 있도록 인덱스 데이터를 변경하는 것이 특징이다.  
RDBMS와 같이 내부적으로 B-Tree로 인덱스를 생성하며, 다중 속성과 고유인덱스를 사용하며 복합 인덱스를 작성할 수 있다.  

<br>
B-Tree
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/NoSQL/Btree1.PNG" height="300" width="600" /></div><br>
출처:Namoosori-MongoDB(ver2.21) PDF  

###  Index 종류
<link rel = "stylesheet" href ="/static/css/bootstrap.min.css">
<table class="table">
	<tbody>
	<tr>
		<td>종류</td><td>사용 환경</td>
	</tr>
	<tr>
		<td>고유 인덱스(Unique Index)</td><td>인덱스의 모든 엔트리가 고유해야 하는 경우</td>
	</tr>
		<tr>
		<td>다중 키 인덱스(Multikey Index)</td><td>인덱스 키로 사용된 필드의 값이 배열인 경우</td>
	</tr>
	<tr>
		<td>공간 정보 인덱스</td><td>좌표 평면으로 적용된다. 다양한 이유로 사용</td>
	</tr>
	</tbody>
</table>
<br>

###  고유 인덱스(Unique Index)
인덱스의 모든 엔트리가 고유해야 하는 경우
<span style ="color: red">**Mongo에서 기본적으로 가지고 있는 _id 또한 고유 인덱스 이다.**</span><br>


sort()를 활용해서 정렬된 자료를 얻을 수 있다. 1: 오름차순 -1: 내림차순
```js
/*
Index: 특정 문서를 탐색 할 때 전부를 탐색하지 않고도 데이터를 찾을 수 있게 한다.
=> 빠르게 데이터를 조회 가능하다.
종류: 고유 인덱스, 다중 키 인덱스, 공간정보 인덱스
*/

//Data설정: user Collection에 id,score field를 가진 30개의 Document 생성
var user ={};
function adduser(){
    for(var i=0;i<30;i++){
    	user.id = i;
    	if(i>0 && i<5 )
    	user.score =3;
    	db.user.insert(user);
    }
}
adduser()

//고유 인덱스(Unique Index): 인덱스의 모든 엔트리가 고유해야 하는 경우
db.user.createIndex({id:1})
/*
sort()를 활용해서 정렬된 자료를 얻을 수 있다.
1: 오름차순
-1: 내림차순
*/
db.user.find().sort({id:1})
db.user.find().sort({id:-1})
```
<br>
결과 - db.user.find().sort({id:1})
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/NoSQL/UniqueIndex1.PNG" height="150" width="600" /></div>
<br>

결과 - db.user.find().sort({id:-1})
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/NoSQL/UniqueIndex2.PNG" height="150" width="600" /></div>
<br>

###  다중키 인덱스(Multikey Index)
인덱스로 사용된 필드의 값이 배열인 경우  
```js
//다중 키 인덱스(Multikey Index): 인덱스 키로 사용된 필드의 값이 배열인 경우
//id: 오름차순, score: 내림차순으로 최적화한 Index 생성
db.user.createIndex({score:1,id:1})
db.user.find().sort({score:1})
```
결과
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/NoSQL/MulIndex1.PNG" height="150" width="600" /></div>
<br>

###  공간정보 인덱스
좌표 평면으로 적용된다. 다양한 이유로 사용된다.
 - Within: 좌표와 Boundary를 지정한 모형을 생성한 후 Boundary안에 있는 Document 찾기  
<link rel = "stylesheet" href ="/static/css/bootstrap.min.css">
<table class="table">
	<tbody>
	<tr>
		<td>종류</td><td>모양</td>
	</tr>
	<tr>
		<td>$box</td><td>사각형</td>
	</tr>
		<tr>
		<td>$center</td><td>원</td>
	</tr>
		<tr>
		<td>$centerSphere</td><td>구</td>
	</tr>
	</tbody>
</table>
<br>
```js
//공간 정보 인덱스 2차원으로 생성
//$geoWithin은 지형 모형을 선택하여 실행하겠다는 의미이다.
//$centerSphere: 구형, $box: 사각형, $center: 원형
db.legacyplaces.find({
  location: {
    $geoWithin: {
      //(x,y)=> Center값 지정후
      //반지름 정하기: radian 을 사용하여 정해지므로 6378.1로 나누었다.
      $centerSphere: [[126.876933, 33.381018], 5 / 6378.1]
    }
  }
})
```
<br>
 - near: 좌표를 지정한 후 가까운 거리 순으로 문서를 찾음  


```js
//$near: 좌표를 지정한 후 가까운 거리 순으로 문서를 찾는 Option
db.places.find({
  location: {
    $nearSphere: {
      $geometry: {
        type: 'Point',
        coordinates: [ 126.941131, 33.459216 ]
      },
      $minDistance: 1000,
      $maxDistance: 12000
    }
  }
})
```
<br>
 - geoNear aggregation stage: aggregation은 쿼리에 포함이 가능하므로 MongoDb aggregation 파이프라인의 장점을 최대한 이용할 수 있다.  


Option
<link rel = "stylesheet" href ="/static/css/bootstrap.min.css">
<table class="table">
	<tbody>
	<tr>
		<td>limit</td><td>가져올 문서의 최대 개수</td>
	</tr>
	<tr>
		<td>query</td><td>문서를 필터링</td>
	</tr>
		<tr>
		<td>near</td><td>기준 좌표점</td>
	</tr>
		<tr>
		<td>distanceField</td><td>거리를 출력 할 필드명</td>
	</tr>
	</tbody>
</table>
<br>
```js
//$geoNear aggregation stage: 가까운 곳을 찾고 거리까지 구하는 방법
db.places.aggregate([
  {
    $geoNear: {
      spherical: true,
      limit: 10,
      maxDistance: 10000,
      near: {
        type: 'Point',
        coordinates: [126.876933, 33.381018]
      },
      distanceField: 'distance',
      key: 'location'
    }
  }
])
```
결과 - Within
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/NoSQL/Geo1.PNG" height="150" width="600" /></div>
<br>
<hr>
내용 참조: <a href="https://docs.mongodb.com/manual/indexes">MongoDB 사이트</a><br>
내용 참조: Namoosori-MongoDB(ver2.21) PDF<br>
내용 참조: <a href="https://blog.ull.im/engineering/2019/03/06/mongodb-geospatial-queries.html">Reid 블로그</a><br>
참조: <a href="https://github.com/wjddyd66/NoSQL/tree/master/Index">원본코드</a><br>
코드에 문제가 있거나 궁금한 점이 있으면 wjddyd66@naver.com으로  Mail을 남겨주세요.