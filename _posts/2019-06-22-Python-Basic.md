---
layout: post
title:  "Python-데이터타입, 연산자"
date:   2019-06-22 06:30:00 +0700
categories: [Python]
---

###  Python 기본 데이터 타입
<link rel = "stylesheet" href ="/static/css/bootstrap.min.css">
<table class="table">
	<tbody>
	<tr>
		<td>타입</td><td>설명</td><td>표현 예</td>
	</tr>
	<tr>
		<td>int</td><td>정수형 데이터</td><td>a=10</td>
	</tr>
		<tr>
		<td>float</td><td>소숫점을 포함한 실수</td><td>a=10.25</td>
	</tr>
		<tr>
		<td>bool</td><td>True or Flase</td><td>a=True</td>
	</tr>
			<tr>
		<td>None</td><td>Null</td><td>a=None</td>
	</tr>
			<tr>
		<td>String</td><td>문자열 데이터</td><td>a="10"</td>
	</tr>
				<tr>
		<td>복소수</td><td>복소수</td><td>a=10+5j</td>
	</tr>
	</tbody>
</table>
<br>
 - Python은 변수의 이름 대문자, 소문자를 구별한다.
 - Python에서 type을 확인하는 함수는 type()으로 확인한다.
 - 복소수는 j를 붙여 허수인 것을 표현하고, real,imag로서 실수와 허수부분을 구별한다.

```python
#기본 데이터 타입
a=10;
print(type(a),a);
a=10.25
print(type(a),a);
a=True;
print(type(a),a);
a=None;
print(type(a),a);
a="10"
print(type(a),a);
#Python은 대소문자를 구별한다.
A=5;
print(type(a),a,type(A),A);
#복소수
a=10+5j;
print("실수부분은 ",a.real,"허수부분은 ",a.imag)
'''
결과
<class 'int'> 10
<class 'float'> 10.25
<class 'bool'> True
<class 'NoneType'> None
<class 'str'> 10
<class 'str'> 10 <class 'int'> 5
실수부분은  10.0 허수부분은  5.0
'''
```
###  Python 문자열
 - 문자열 포맷팅: 일정한 포맷에 맞춰 문자열을 조합하는 것 이다.  
%d와 같이 자리를 잡고 나중에 값을 대입하는 방식이다.  
%d와 같이 자리를 잡는 지시어는 Conversion Specifier라 한다.

<table class="table">
	<tbody>
	<tr>
		<td>Conversion Specifier</td><td>의미</td>
	</tr>
	<tr>
		<td>%s</td><td>문자열</td>
	</tr>
		<tr>
		<td>%c</td><td>문자</td>
	</tr>
		<tr>
		<td>%d, %i</td><td>정수</td>
	</tr>
			<tr>
		<td>%f, %F</td><td>부동소수</td>
	</tr>
			<tr>
		<td>%e, %E</td><td>지수형 부동소수</td>
	</tr>
				<tr>
		<td>%g, %G</td><td>일반형</td>
	</tr>
				<tr>
		<td>%o, %O</td><td>8진수</td>
	</tr>
				<tr>
		<td>%x, %X</td><td>16진수</td>
	</tr>
				<tr>
		<td>%%</td><td>%퍼센트 리터럴</td>
	</tr>
	</tbody>
</table>
<br>
```python
#문자열 포멧팅
ss="이름: %s 나이: %d"%("황정용",26)
print(ss);#이름: 황정용 나이: 26
```
<br>
 - 문자열 메서드: 문자열 str 클래스에서 여러가지 유용한 메서드를 제공하고 있다.
<table class="table">
	<tbody>
	<tr>
		<td>Method</td><td>사용 이유</td>
	</tr>
	<tr>
		<td>str.join()</td><td>여러개의 문자열을 하나로 결합</td>
	</tr>
		<tr>
		<td>str.split()</td><td>문자열을 분리 후 List형태로 Return</td>
	</tr>
		<tr>
		<td>str.partition()</td><td>앞부분, 분리자, 뒷부분 의 3개의 값으로 문자열 분리하여 Return</td>
	</tr>
			<tr>
		<td>str.format()</td><td>문자열 포멧팅</td>
	</tr>
	</tbody>
</table>
<br>
```python
#문자열 메소드
#str.join()
s = ','.join(['황정용',"26",'Programmer']);
# ',': 문자열을 합칠때 넣어주는 문자열
print(s);#황정용,26,Programmer

#str.split()
s = '황정용,26,Programmer'.split(',');
print(type(s),s); #<class 'list'> ['황정용', '26', 'Programmer']

#str.partition()
s1,s2,s3 = '황정용,26,Programmer'.partition(',');
print(type(s1),s1);
print(type(s2),s2);
print(type(s3),s3);
'''
<class 'str'> 황정용
<class 'str'> ,
<class 'str'> 26,Programmer
'''

#str.format()
s="Name:{0},Age:{1},Job:{2}".format('황정용','26','Programmer')
print(s);#Name:황정용,Age:26,Job:Programmer
```
###  연산자
<table class="table">
	<tbody>
	<tr>
		<td>연산자</td><td>사용 이유</td><td>종류</td>
	</tr>
	<tr>
		<td>산술연산자</td><td>값 계산</td><td>+,-,*,/,**,%,//등</td>
	</tr>
		<tr>
		<td>비교 연산자</td><td>값 비교</td><td>
		<,
		>,
		==,!=등
		</td>
	</tr>
		<tr>
		<td>할당연산자</td><td>변수에 값할당</td><td>+=,-=,*=,%=,//=등</td>
	</tr>
			<tr>
		<td>논리연산자</td><td>조건 판단</td><td>and, or, not</td>
	</tr>
		<tr>
		<td>Bitwise 연산자</td><td>비트 단위 연산</td><td>&, |, ^, ~, <
		<,
        >>, 
		</td>
	</tr>
		<tr>
		<td>멤버쉽 연산자</td><td>왼쪽값이 오른쪽 안에 포함되어있나 확인</td><td>in, not in</td>
	</tr>
		<tr>
		<td>identity 연산자</td><td>양쪽 값이 동일한 Object를 가르키는지 확인</td><td>is, is not</td>
	</tr>
	</tbody>
</table>
<br>
<span style ="color: red">**id(): 객체의 주소를 확인**</span><br>
<span style ="color: red">**is: 객체의 주소를 비교**</span><br>
<span style ="color: red">**==: 객체의 값을 비교**</span><br>
```python
#연산자

#산술 연산자
a=5;
print(a%2); #1
print(a/2); #2.5

#비교 연산자
if a ==5:
    print('참');
else:
    print('거짓');
#참

#할당 연산자
a *=10
print(a)#50

#논리 연산자
if a==50 and a<100:
    print('참');
else:
    print('거짓');
#참

#Bitwise 연산자
a=10;
b=11;
c = a&b; #&: and, |: or
d = a^b; #^: XOR
print(c) # 10
print(d) # 1

#멤버쉽 연산자
a = [1,2,3,4]
b= 1 in a
print(b) # True

#Identity 연산자
a = "ABC"
b=a;
print(a is b);#True
```
<br>
<hr>
내용참조:<a href="http://pythonstudy.xyz/python/article/9-%EB%AC%B8%EC%9E%90%EC%97%B4%EA%B3%BC-%EB%B0%94%EC%9D%B4%ED%8A%B8">예제로 배우는 Python 프로그래밍</a><br>
내용참조:<a href="http://pythonstudy.xyz/python/article/8-%EC%97%B0%EC%82%B0%EC%9E%90">예제로 배우는 Python 프로그래밍</a><br>
참조:<a href="https://github.com/wjddyd66/Python/tree/master/Basic">원본코드</a><br>
코드에 문제가 있거나 궁금한 점이 있으면 wjddyd66@naver.com으로  Mail을 남겨주세요.