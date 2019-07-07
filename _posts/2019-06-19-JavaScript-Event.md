---
layout: post
title:  "JavaScript-Event"
date:   2019-06-19 11:30:00 +0700
categories: [Web]
---

###  Event

Event: 모든 행위를 말하는 것으로 프로그램에서는 미리 사용자의 행위를 예측하여 미리 사용할 수 있도록 많은 Event를 준비해 놓는다.  

EvnetHandler: Event와 준비한 프로그램을 연결해주는 역할을 한다.  

 

###  Event 등록
Event를 등록하는 방법은 크게 두가지가 있다.  
1. 태그에 직접 지정
2. Script태그 안에서 지정  

아래 실행 화면은 Button1: Tag, Button2: Script 에서 Event를 등록한 것이다.  

<iframe width="100%" height="350" src="//jsfiddle.net/wjddyd66/t3h156m8/8/embedded/html,js,result/dark/" allowfullscreen="allowfullscreen" frameborder="0"></iframe>
<br>

###  이벤트 버블링(Event Bubbling)
이벤트 버블링이란 엘레멘트에서 이벤트가 감지 되었을 때, 해당 엘리먼트를 포함하고 있는 부모 엘리먼트를 통하여 최상위 까지 이벤트가 전달되는 것이다.  
아래 화면은 글자 P를 누르게 되면 p -> div -> form 순으로 이벤트가 전달되는 것을 알 수 있다.  
<iframe width="100%" height="350" src="//jsfiddle.net/wjddyd66/t3h156m8/10/embedded/html,result/dark/" allowfullscreen="allowfullscreen" frameborder="0"></iframe>
<br>

###  이벤트 캡쳐(Event Capture)
이벤트 캡쳐는 이벤트 버블링과 반대이다.  
최초 이벤트가 발생한 자식 요소로 내려가는 과정을 의미한다.  
<iframe width="100%" height="350" src="//jsfiddle.net/wjddyd66/t3h156m8/30/embedded/html,js/dark/" allowfullscreen="allowfullscreen" frameborder="0"></iframe>
<br>
결과:  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/JavaScript/Js32.JPG" height="150" width="600" /></div><br>

###  이벤트 위임(Event Delegation)
이벤트 위임을 통해 다수의 요소에 공통으로 적용되는 핸들러를 공통된 조상 요소에 단 한번만 열결하면 동작 할 수 있도록 해준다.  
ul Element 에 Event를 걸어 자식인 a 링크가 가지 않게 되는 실습이다.  
작동되는 a 링크 - 이벤트X:  
<iframe width="100%" height="350" src="//jsfiddle.net/wjddyd66/t3h156m8/39/embedded/html,result/dark/" allowfullscreen="allowfullscreen" frameborder="0"></iframe>
<br>

작동되지 않는 a 링크 - 이벤트X:  
<iframe width="100%" height="350" src="//jsfiddle.net/wjddyd66/t3h156m8/50/embedded/html,js,result/dark/" allowfullscreen="allowfullscreen" frameborder="0"></iframe>
<br>

###  이벤트 예제 달력만들기
<iframe width="100%" height="350" src="//jsfiddle.net/wjddyd66/n8tw9j2p/embedded/html,result/dark/" allowfullscreen="allowfullscreen" frameborder="0"></iframe>
<br>

<hr>
내용참조:<a href="https://joshua1988.github.io/web-development/javascript/event-propagation-delegation/#%EC%9D%B4%EB%B2%A4%ED%8A%B8-%EC%9C%84%EC%9E%84---event-delegation">Captain Pangyo 블로그</a><br>
내용참조:<a href="http://tcpschool.com/jquery/jq_event_delegation">TCP School</a><br>
코드에 문제가 있거나 궁금한 점이 있으면 wjddyd66@naver.com으로  Mail을 남겨주세요.