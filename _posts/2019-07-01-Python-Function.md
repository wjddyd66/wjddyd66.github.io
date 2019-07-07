---
layout: post
title:  "Python-Function"
date:   2019-07-01 07:00:00 +0700
categories: [Python]
---

###  Python 함수
Python은 함수 지향적 언어로서 함수의 종류는 크게 2가지로 나뉘게 된다.  
1. 내장함수: 이미 지정되어있는 함수
2. 사용자 정의함수: def 키워드로 사용자가 직접 작성하는 함수

Python은 함수 안에 함수가 정의될 수 있다.  
```python
#Function

#내장함수: 이미 지정되어있는 함수
import math
print(math.ceil(1.2), math.ceil(1.7)) #2 2 ceil(): 올림
print(math.floor(1.2), math.floor(1.7)) #1 1 floor(): 내림

#사용자 정의함수: def 키워드를 사용하여 정의한다.
def DoFunc1():
    print("DoFunc1 처리")
DoFunc1()    

#파이썬 함수에서 입력 파라미터는 Pass by Assignment에 의해 전달된다. 
#즉, 호출자(Caller)는 입력 파라미터 객체에 대해 레퍼런스를 생성하여 레퍼런스 값을 복사하여 전달
def DoFunc2(arg1, arg2):
    DoFunc3()
    return arg1+arg2

def DoFunc3():
    print("함수가 함수를 호출 가능")
    
aa = DoFunc2(1,2)
print(aa)   
aa = DoFunc2("대한","민국")
print(aa)   
print("DoFunc2의 객체주소: ", id(DoFunc2))
'''
DoFunc1 처리
함수가 함수를 호출 가능
3
함수가 함수를 호출 가능
대한민국
DoFunc2의 객체주소:  2011568618488
'''

```
<br>

###  Python Parameter
<link rel = "stylesheet" href ="/static/css/bootstrap.min.css">
<table class="table">
	<tbody>
	<tr>
		<td>Parameter</td><td>순서에 맞게 값을 전달</td>
	</tr>
	<tr>
		<td>Default Parameter</td><td>입력파라미터 중 호출자가 전달되지 않으면 Default로 지정된 값을 사용할 수 있다.</td>
	</tr>
		<tr>
		<td>Named Parameter</td><td>Parameter에 이름을 주어서 파라미터의 순서에 상관없이 값을 줄 수 있다. => 가독성이 높아진다.</td>
	</tr>
		<tr>
		<td>가변길이 파라미터</td><td>0~N개의 파라미터를 받아들일 수 있는 표현방법(*: List, **:Dict)</td>
	</tr>

	</tbody>
</table>
<br>


```python
#Default Parameter: 입력파라미터 중 호출자가 전달되지 않으면
#Default로 지정된 값을 사용할 수 있다.
def calc(i, j, factor = 1):
    return i * j * factor
 
result = calc(10, 20)
print(result) #200 factor가 1로서 Default값이 들어감

#Named Parameter: Parameter에 이름을 주어서 파라미터의 순서에 상관없이
#값을 줄 수 있다. => 가독성이 높아진다.
def report(name, age, score):
    print(name, score)
 
report(age=10, name="Kim", score=80) #Kim 80

#가변길이 파라미터: 0~N개의 파라미터를 받아들일 수 있는 표현방법
# *: List, **:Dict이다. 

#List 받기
def test_var_args(f_arg, *args):
    print("first normal arg:", f_arg)
    for arg in args:
        print("another arg through *argv:", arg)

test_var_args('yasoob', 'python', 'eggs', 'test')
'''
first normal arg: yasoob
another arg through *argv: python
another arg through *argv: eggs
another arg through *argv: test
'''
#Dict 받기
def greet_me(**kwargs):
    print(kwargs.items())
    for key, value in kwargs.items():
        print("{0} = {1}".format(key, value))
        
greet_me(name="yasoob", school="snu")
'''
dict_items([('name', 'yasoob'), ('school', 'snu')])
name = yasoob
school = snu
'''

```

###  Python 변수의 생존범위(Scope Rule)
변수 영역 및 접근 순서: 1) Local 2)Enclosing Function 3) Global 4) Built-in  

<table class="table">
	<tbody>
	<tr>
		<td>Local</td><td>가장 가까운 함수안 범위</td>
	</tr>
	<tr>
		<td>Enclosing Function</td><td>가장 가까운 함수가 아닌 두번째 이상의 함수 가까운 함수범위</td>
	</tr>
		<tr>
		<td>Global</td><td>함수 바깥의 변수 또는 import된 module</td>
	</tr>
		<tr>
		<td>Built-in</td><td>파이썬안에 내장되어 있는 함수 또는 속성들</td>
	</tr>

	</tbody>
</table>
<br>


```python
#변수의 생존범위: Scope Rule
#변수 영역 및 접근 순서: 1) Local 2)Enclosing Function 3) Global 4) Built-in
a=10; b=20; c=30
print("실행1) a:{}, b:{}, c:{}".format(a, b, c))
def func1():
    a=40
    b=50
    def func2():
            global c
            nonlocal b
            print("실행2) a:{}, b:{}, c:{}".format(a, b, c))   
            c=60
            b=70
    func2()
    print("실행3) a:{}, b:{}, c:{}".format(a, b, c))   
func1()
print("실행4) a:{}, b:{}, c:{}".format(a, b, c))  
'''
실행1) a:10, b:20, c:30
실행2) a:40, b:50, c:30
실행3) a:40, b:70, c:60
실행4) a:10, b:20, c:60
'''

```
<br>
<hr>
내용참조:<a href="https://suwoni-codelab.com/python%20%EA%B8%B0%EB%B3%B8/2018/03/05/Python-Basic-scope">Suwoni-Codelab</a><br>
내용참조:<a href="https://blog.naver.com/PostView.nhn?blogId=mint3081&logNo=221537426126&parentCategoryNo=&categoryNo=49&viewDate=&isShowPopularPosts=false&from=postList">천프로 블로그</a><br>
참조:<a href="https://github.com/wjddyd66/Python/tree/master/Function">원본코드</a><br>
코드에 문제가 있거나 궁금한 점이 있으면 wjddyd66@naver.com으로  Mail을 남겨주세요.