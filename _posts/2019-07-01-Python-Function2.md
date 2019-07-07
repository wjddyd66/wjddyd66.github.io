---
layout: post
title:  "Python-Function2(클로저, 일급함수, 람다, 함수장식자,재귀함수)"
date:   2019-07-01 07:00:00 +0700
categories: [Python]
---

###  클로저
클로저란 내부함사의 객체를 반환한다.  
이러한 특성으로 인하여 함수 지향적인 언어인 Python을 객체지향의 특성을 따라할 수 있게 된다.  

```python
'''
클로져: 내부함수의 객체를 반환한다.
scope에 제약을 받지않는 변수를 포함하는 코드 블럭이다.
참고: 함수의 이름은 객체의 주소를 참조한다.
함수 지향적인 언어이나 객체지향의 특성을 따라하려고 한 개념이다.
'''
def Outer():
    count = 0
    def Inner():
        nonlocal count
        count += 1
        return count
    print(Inner())
Outer()   #1

def Outer2():
    count = 0
    def Inner2():
        nonlocal count
        count += 1
        return count
    return Inner2 # <=== closure: 내부함수의 주소를 반환

aaa = Outer2() # Inner2의 주소를 기억한다.
print(aaa)
print(aaa())
print(aaa())
print(aaa())
'''
<function Outer2.<locals>.Inner2 at 0x00000282234E3F28>
1
2
3
'''
bbb = Outer2() # Inner2의 주소를 기억한다.
print(bbb)
print(bbb())
print(bbb())
'''
<function Outer2.<locals>.Inner2 at 0x000002621B724048>
1
2
같은 함수를 참조하고 있으나 다른 인스턴스를 가지고 있다.
'''

```
<br>

###  일급함수(first-class)
일급 함수(first-class)는 객체 지향 프로그래밍(object-oriented programming) 중에서 파이썬을 포함한 몇몇 프로그래밍 언어에서 발견할 수 있는 개념  
1. 함수를 변수에 할당 가능
2. 다른 함수에서 해당 함수를 인자로 전달 가능
3. 함수에서 함수를 반환 가능


```python
'''
일급 함수(first-class)는 객체 지향 프로그래밍(object-oriented programming) 
중에서 파이썬을 포함한 몇몇 프로그래밍 언어에서 발견할 수 있는 개념
1. 함수를 변수에 할당 가능
2. 다른 함수에서 해당 함수를 인자로 전달 가능
3. 함수에서 함수를 반환 가능
'''
def fnc1(a, b):
    return a+b

fnc2 = fnc1
print(fnc1(3, 4)) #7
print(fnc2(3, 4)) #7

print()
def fnc3(fnc):
    def fnc4():
        print("나는 내부함수")
    fnc4()
    return fnc    

mbc =  fnc3(fnc1) #인자로 fnc1의 주소를 넘긴다.
print(mbc(3, 4)) #나는 내부함수 7

```

###  람다(Lambda, 축약함수)
function을 한줄로 작성 할 수 있는 작성 방식  
함수를 여러줄에 걸쳐서 사용하게 되면 가독성이 떨어지고 만들기도 어렵기 때문에 람다 사용  
람다의 결과를 변수가 받을 수도 있다.  


```python
'''
람다(Lambda, 축약함수)
function을 한줄로 작성 할 수 있는 작성 방식
함수를 여러줄에 걸쳐서 사용하게 되면 가독성이 떨어지고 만들기도 어렵기 때문에 람다 사용
람다의 결과를 변수가 받을 수도 있다.
'''
#일반함수
def Hap(x, y):
    return x+y
print(Hap(1, 2)) #3
#람다
print((lambda x, y : x+y)(1, 2)) #3

```
<br>

###  함수장식자(Decorator)
다른 함수를 감싼 함수. Meta의 기능이 있다.  
@로서 표현하게 된다.  



```python
'''
함수장식자(Decorator)
다른 함수를 감싼 함수. meta의 기능이 있다.
'''
#함수 장식자 사용X
def make2(fn):
    return lambda : "Hello" + fn()

def make1(fn):
    return lambda : "World" + fn()

def hello():
    return "황정용"

hi = make2(make1(hello))
print(hi()) #HelloWorld황정용

#함수 장식자 사용
@make2
@make1
def hello2():
    return "황정용2"
print(hello2()) #HelloWorld황정용2
```
<br>

###  재귀함수(Recursive Call)
계속해서 함수 자기자신을 호출하여 반복문과 같은 효과를 낼 수 있다.  
계속 반복되므로 탈출 조건을 명시하여야 한다.  



```python
'''
재귀함수(Recursive Call)
계속해서 함수 자기자신을 호출하여 반복문과 같은 효과를 낼 수 있다.
계속 반복되므로 탈출 조건을 명시하여야 한다.
'''
def countDown(n):
    if n == 0 :
        print("완료")
    else :
        print(n, end = " ")
        countDown(n-1) #자기자신을 호출
countDown(5) #5 4 3 2 1 완료

```
<br>
<hr>
참조:<a href="https://github.com/wjddyd66/Python/tree/master/Function">원본코드</a><br>
코드에 문제가 있거나 궁금한 점이 있으면 wjddyd66@naver.com으로  Mail을 남겨주세요.