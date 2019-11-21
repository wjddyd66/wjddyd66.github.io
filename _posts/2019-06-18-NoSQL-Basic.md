---
layout: post
title:  "NoSQL 개념"
date:   2019-06-18 12:00:00 +0700
categories: [NoSQL]
---

<span style ="color: red">본 글은 NoSQL Post는 Namoosori-MongoDB(ver2.21) PDF자료를 참조하여 정리하였습니다.</span>  
###  NoSQL
NoSQL(Not Only SQL)은 RDBMS의 한계를 극복하기 위해 만들어진 새로운 형태의 데이터 베이스 이다. RDBMS와 같이 고정된 스키마가 없이 Key-Value형식으로 이루어진 거대한 Map 이다.  
<span style ="color: red">RDBMS의 한계</span><br>
1. Big Data를 스키마에 맞게 변경하는데 시간이 오래 걸린다.
2. RDBMS이 지원하는 "일관성"을 유지하는 것은 분산환경에서 맞지 않는다.

###  NoSQL 저장소
1. Key/Value Database : <span style ="color: red">Redis(인메모리)</span>, Oracle Coherence 등
 - 단순한 저장구조를 가지며, 복잡한 조회 연산을 지원하지 않는다.
 - 고속 읽기와 쓰기에 최적화된 경우가 많다. key에 대한 단위 연산이 빠른 것이지, 여러 key에 대한 연산은 느릴 수 있다.
 - 메모리를 저장소로 쓰는 경우, 아주 빠른 get과 put을 지원한다.
 - Value는 문자열이나 정수와 같은 원시 타입이 들어갈 수도 있고, 아래 사진처럼 또 다른 key/Value이 들어갈 수도 있다. 이를 Column Family이라고 하며, Key 안에 (Column, Value) 조합으로 된 여러 개의 필드를 갖는 것을 말한다.

2. Big Table Database (= Ordered Key/Value) : <span style ="color: red">Apache Cassandra</span>,Hbase등
 - Key/Value Store와 데이터 저장 방식은 동일하다. 보통 NoSQL은 RDBMS의 order by같은 정렬기능을 제공해주지 않는다. 그러나 이 모델은 내부적으로 Key를 정렬한다.
 - 키를 정렬함으로써, 값을 날짜나 선착순으로 정렬해서 보여줄 때 매우 유용하다.

3. Document Database : <span style ="color: red">MongoDB</span>, CouchDB, Riak
 - Key/Value Store의 확장된 형태로, value에 Document라는 타입을 저장한다. Document는 구조화된 문서 데이터(XML, JSON, YAML 등)을 말한다.
 - 복잡한 데이터 구조를 표현가능하다.
 - Document id 또는 특정 속성값 기준으로 인덱스를 생성한다. 이 경우 해당 key 값의 range에 대한 효율적인 연산이 가능해 지므로 이에 대한 쿼리를 제공한다. 따라서 Sorting, Join, Grouping 등이 가능해진다.
 - 쿼리 처리에 있어서 데이터를 파싱해서 연산을 해야하므로 overhead가 key-value 모델보다 크다. 큰 크기의 document를 다룰 때는 성능이 저하된다.

4. Graph Database : <span style ="color: red">neo4j</span>,Sones, AllegroGraph
 - node들과 relationship들로 구성된 개념이다. 역시 Key/Value Store이며 모든 노드는 끊기지 않고 연결되어 있다.
 - relationship은 direction, type, start node, end node에 대한 속성 등을 가진다. 보통 양(코스트, 무게 등)적인 속성들을 가진다.

###  CAP
NoSQL은 네트워크 상에서의 분산시스템이다. 이러한 신뢰할 수 없는 네트워크 분산 시스템을 다룰 때, 일치성과 가용성을 고려해야 한다.  
1. 일관성(Consistency) : 분산된 노드 중 어느 노드로 접근하더라도 데이터 값이 같아야 한다. (데이터 복제 중에 쿼리가 되는 일관성을 제공하지 않는 시스템의 경우 다른 데이터 값이 쿼리될 수 있다.)
2. 가용성(Availability) : 클러스터링된 노드 중 하나 이상의 노드가 실패(Fail)라도 정상적으로 요청을 처리할 수 있는 기능을 제공한다.
3. 분산 허용(Partition Tolerance): 클러스터링 노드 간에 통신하는 네트워크가 장애가 나더라도 정상적으로 서비스를 수행한다. 노드 간 물리적으로 전혀 다른 네트워크공간에 위치도 가능하다.

###  DB 샤딩
정보를 적절한 크기로 나누는 메커니즘이다.  NoSQL 시스템은 다운 시간 없이 수용하고 분리하는 작업이 매우 중요하다.  이로 인하여 DB 클러스트를 자연스럽게 확장하게 해준다.  
샤딩의 기준
1. 이름 기준
2. 지역기준
3. 무작위

<hr>
내용 참조:<a href="https://sjh836.tistory.com/97">sjh836 블로그</a><br>
내용 참조: Namoosori-MongoDB(ver2.21) PDF<br>
코드에 문제가 있거나 궁금한 점이 있으면 wjddyd66@naver.com으로  Mail을 남겨주세요.