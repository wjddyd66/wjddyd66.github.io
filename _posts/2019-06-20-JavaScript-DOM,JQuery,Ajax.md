---
layout: post
title:  "JavaScript-DOM,JQuery,Ajax"
date:   2019-06-20 08:00:00 +0700
categories: [Web]
---

###  DOM

DOM이란 브라우저 화면에 보이는 요소를 조작하기 위한 기능을 모은 라이브러리 집합 이다.  
<link rel = "stylesheet" href ="/static/css/bootstrap.min.css">
<table class="table">
	<tbody>
	<tr>
		<td>D(Document)</td><td>문서 객체 모델: 작성된 웹 문서가 객체로 인식</td>
	</tr>
	<tr>
		<td>O(Object)</td>
		<td>
		<ul>
		<li>사용자 정의 객체</li>
		<li>정의된 내장 객체</li>
		<li>웹브라우저 객체</li>
		</ul>
		</td>
	</tr>
		<tr>
		<td>M(Model)</td><td>Html은 Element 로 이루어져서 <
		>...<
		/>
        로 이루어 진다 이런 Element안에 다른 Element를 추가하여 Tree형식으로 구성되어 
        부모/자식 관계로 접근할 수 있다.

		</td>
	</tr>
	
	</tbody>
</table>
아래 예제는 DOM을 이해하기 위한 구구단 출력 예제 이다.
<br>
<iframe width="100%" height="350" src="//jsfiddle.net/wjddyd66/te8wmjgh/9/embedded/html,js,result/dark/" allowfullscreen="allowfullscreen" frameborder="0"></iframe>
<br>

###  JQuery
JQuery란 자바 스크립트 라이브러리로 JavaScript를 좀더 쉽게 사용하기 위하여 사용한다.  
DOM 형식의 JavaScript: documnet.getElementById("ID")  
DOM 형식의 JQuery: $("#ID") (#: ID, .: Class)  
사용방법  
1. CDN 호스트를 통해 JQuery를 불러오는 방법
 - <
 script src="url">
2. 파일을 설치하여 사용하는 방법  
 - <
 script src="다운로드 경로">

<span style ="color: red">**Chain: JQuery는 반환값으로 자기 자신을 반환하는 특징을 가지고 있다. 이러한 특징을 사용하여 한번 선택한 대상에 대해서 연속적인 제어가 가능하다.**</span><br>
<span style ="color: red">**Chain을 사용하면 코드가 간결해지고 DOM을 활용하여 Element에 쉽게 접근할 수 있는 장점을 가지고 있다.**</span><br>

아래 예제는 JQuery을 이해하기 위한 계산기 예제 이다.
<br>
<iframe width="100%" height="350" src="//jsfiddle.net/wjddyd66/te8wmjgh/58/embedded/html,js,result/dark/" allowfullscreen="allowfullscreen" frameborder="0"></iframe>
<br>

###  Ajax(Asynchronous Javascript And Xml)
Ajax란 JavaScript를 사용한 비동기 통신이다.  
HTTP 프로토콜은 Client -> Server: Request보냄, Server -> Client: Response보냄 으로 이루어 진다.  
이로 인하여 페이지를 전체를 다시 로드해야 하는 시간 낭비가 발생한다.  
이러한 HTTP프로토콜에서 일부 페이지만 갱신하기 위하여 사용하는 것이 Ajax이다.  
<span style ="color: red">**Ajax가 하나의 Thread라고 생각하면 된다.**</span><br>
장점
1. 웹페이지 속도 향상
2. 서버와 다른 Thread에서 작동하므로 서버 처리가 완료될때 까지 기다리지 않아도 된다.
3. 기존에 웹에서 불가능했던 다양한 UI를 지원할 수 있다.

단점
1. 연속적으로 데이터 요청 시 서버 부하가 증가할 수 있다.
2. 사용자에게 아무런 진행 정보가 주어지지 않는다.

아래 예시는 Ajax 통신 성공과 실패에 대한 Code이다.  
<iframe width="100%" height="350" src="//jsfiddle.net/wjddyd66/utc2k4er/1/embedded/html,js,css,result/dark/" allowfullscreen="allowfullscreen" frameborder="0"></iframe>
<br>

<hr>
내용참조:<a href="https://webclub.tistory.com/218">Web Club 블로그</a><br>
내용참조:<a href="http://tcpschool.com/jquery/jq_event_delegation">TCP School</a><br>
내용참조:<a href="https://coding-factory.tistory.com/143">코딩팩토리 블로그</a><br>
코드에 문제가 있거나 궁금한 점이 있으면 wjddyd66@naver.com으로  Mail을 남겨주세요.