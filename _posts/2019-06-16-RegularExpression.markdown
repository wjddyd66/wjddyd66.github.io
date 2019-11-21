---
layout: post
title:  "정규 표현식"
date:   2019-06-16 09:30:20 +0700
categories: [others]
---

###  정규 표현
Input의 형태를 강제로 정해주기 위하여 사용한다.  

###  문법
<link rel = "stylesheet" href ="/static/css/bootstrap.min.css">
<br>

<table class = "table">
  <thead>
    <tr>
      <th width="100">표현 식</th>
      <th>설명</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>^</td>
      <td>문자열 시작</td>
    </tr>
    <tr>
      <td>$</td>
      <td>문자열 종료</td>
    </tr>
    <tr>
      <td>.</td>
      <td>임의의 문자 [단 ‘'는 넣을 수 없습니다.]</td>
    </tr>
    <tr>
      <td>*</td>
      <td>앞 문자가 0개 이상의 개수가 존재할 수 있습니다.</td>
    </tr>
    <tr>
      <td>+</td>
      <td>앞 문자가 1개 이상의 개수가 존재할 수 있습니다.</td>
    </tr>
    <tr>
      <td>?</td>
      <td>앞 문자가 없거나 하나 있을 수 있습니다.</td>
    </tr>
    <tr>
      <td>[]</td>
      <td>문자의 집합이나 범위를 표현합니다. -기호를 통해 범위를 나타낼 수 있습니다. ^가 존재하면 not을 나타냅니다.</td>
    </tr>
    <tr>
      <td>{}</td>
      <td>횟수 또는 범위를 나타냅니다.</td>
    </tr>
    <tr>
      <td>()</td>
      <td>괄호안의 문자를 하나의 문자로 인식합니다.</td>
    </tr>
    <tr>
      <td>|</td>
      <td>패턴을 OR 연산을 수행할 때 사용합니다.</td>
    </tr>
    <tr>
      <td>\s</td>
      <td>공백 문자</td>
    </tr>
    <tr>
      <td>\S</td>
      <td>공백 문자가 아닌 나머지 문자</td>
    </tr>
    <tr>
      <td>\w</td>
      <td>알파벳이나 문자</td>
    </tr>
    <tr>
      <td>\W</td>
      <td>알파벳이나 숫자를 제외한 문자</td>
    </tr>
    <tr>
      <td>\d</td>
      <td>[0-9] 숫자</td>
    </tr>
    <tr>
      <td>\D</td>
      <td>숫자를 제외한 모든 문자</td>
    </tr>
    <tr>
      <td>(?i)</td>
      <td>대소문자를 구분하지 않습니다.</td>
    </tr>
  </tbody>
</table>

<br>

###  예시
 - 소문자, 숫자, 특수문자(._-) 포함

 - @

 - 소문자와.2글자 이상 6글자 이하

    

=>/^([a-z0-9_\.-]+)@([\da-z\.-]+)\.([a-z\.]{2,6})$/

<hr>
참조: <a href="https://nesoy.github.io/articles/2018-06/Java-RegExp">원본코드</a><br>
코드에 문제가 있거나 궁금한 점이 있으면 wjddyd66@naver.com으로  Mail을 남겨주세요.