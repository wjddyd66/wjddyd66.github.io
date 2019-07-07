---
layout: post
title:  "Python-Class"
date:   2019-07-01 08:00:00 +0700
categories: [Python]
---

###  Class
Class란 객체를 만들기 위하여 정의한다.  
여러가지의 공통된 것을 정의하기 위하여 사용한다.  
각각의 객체들은 원형 클래스를 참조하는 방식이다.  
class '이름' 으로서 선언하게 된다.  
<a href="https://wjddyd66.github.io/java/2019/06/14/Class_Method.html">Class 내용</a><br>
<span style ="color: red">**자바와의 차이**</span><br>
<span style ="color: red">**1) 메소드 오버로딩, 생성자 오버로딩X(메소드 오버로딩 O)**</span><br>
<span style ="color: red">**2) 접근 지정자X**</span><br>
<span style ="color: red">**3) this 대신 self라는 키워드로 비슷한 역할 수행**</span><br>

###  생성자, 소멸자
<link rel = "stylesheet" href ="/static/css/bootstrap.min.css">
<table class="table">
	<tbody>
	<tr>
		<td>생성자</td><td>객체가 생성될때 실행</td><td>__init__</td>
	</tr>
	<tr>
		<td>소멸자</td><td>객체가 소멸할때 실행</td><td>__del__</td>
	</tr>

	</tbody>
</table>
<br>
```python
#Class
def func():
    print("함수입니다.")
    
class TestClass:
    aa = 1 #멤버변수 (전역)
    
    def __init__(self):
        print("생성자")
        
    def __del__(self):
        print("소멸자")    
        
    def myMethod(self):
        name = "tom" #지역변수
        print(name)
        print(self.aa) #클래스 내의 멤버를 호출할 때는 self를 통해 호출.
        
    def abc(self):    
        self.myMethod()
        
test = TestClass() #생성자 호출(init 호출). 객체(instance) 생성
print(test.aa)
test.myMethod() #Bound Method Call
#객체변수가 알아서 아규먼트를 타고 들어간다.

print()
print(TestClass.aa) #원형클래스의 멤버 호출
TestClass.myMethod(test) #Unbound Method Call
#바운드 메소드 콜과는 다르게, 객체변수를 아규먼트로 직접 주어야한다.

print()
print(type(1))
print(type(1.5))
print(type(test)) #type: TestClass

print()
print(id(TestClass))
print(id(test))
'''
생성자
1
tom
1

1
tom
1

<class 'int'>
<class 'float'>
<class '__main__.TestClass'>

1856097037400
1856129974512
소멸자
'''
```
<br>

###  인스턴스, 원형 클래스
<span style ="color: red">**원형 클래스**</span>모든 해당 클래스의 인스턴스가 참조할 수 있는 공유의 영역<br>
<span style ="color: red">**인스턴스**</span>원형 클래스를 참조하여 만들어지는 것. 나중에 멤버(원형클래스에 없는 속성)를 추가하더라도 동적으로 처리된다.<br>

###  Method Call
Method Call은 두가지 종류가 있다.  
1. unbound method call: 클래스를 통해 함수를 호출하며, 인스턴스 객체를 parameter로 전달
2. bound method call: 인스턴스 객체에 bind된 함수를 호출


```python
#Class Car 선언
class Car:
    handle = 0
    speed = 0
    
    def __init__(self, name, speed):
        self.name = name
        self.speed = speed
        
    def showData(self):
        km = "킬로미터" 
        msg = "속도: " + str(self.speed) +km
        return msg

#Car1 인스턴스 생성
car1 = Car("tom", 10)
#Car1 속성 추가
car1.color="검정"

print(car1.handle, " ", car1.name, " ", car1.speed) #0   tom   10
print("car1.color: ", car1.color) #car1.color:  검정

#Car2 인스턴스 생성
car2 = Car("jamez", 20)
print(car2.handle, " ", car2.name, " ", car2.speed) #0   jamez   20

print("주소: ", Car, car1, car2)
print("주소: ", id(Car), id(car1), id(car2))
print("각 객체멤버: ", car1.__dict__)
print("각 객체멤버: ", car2.__dict__)
'''
주소:  <class '__main__.Car'> <__main__.Car object at 0x00000218F82C3278> <__main__.Car object at 0x00000218F82C3240>
주소:  2306231774296 2306266116728 2306266116672
각 객체멤버:  {'name': 'tom', 'speed': 10, 'color': '검정'}
각 객체멤버:  {'name': 'jamez', 'speed': 20}
'''

#Bound Method Call
print(car1.showData()) #속도: 10킬로미터
#Unbound Method Call
print(Car.showData(car1)) #속도: 10킬로미터

#소멸자
```
<br>

###  클래스의 상속
상속: 어떤클래스를 만들 때 다른 클래스의 기능을 물려받을 수 있는 것  
기존 클래스가 라이브러리 형태로 제공되거나 수정이 허용되지 않는 상황이라면 상속을 사용하여야 한다.  


```python
#PohamHandle.py
#Class 선언
class PohamHandle:
    quantity = 0
    
    def leftTurn(self, quantity):
        self.quantity = quantity
        return "좌회전"
    
    def rightTurn(self, quantity):
        self.quantity = quantity
        return "우회전"

#PohamCar.py
#Class 상속
from pack.PohamHandle import PohamHandle

class PohamCar:
    turnShow = "정지"
    
    def __init__(self, ownerName):
        self.ownerName = ownerName
        self.handle = PohamHandle() #클래스의 포함관계
        
    def TurnHandle(self, q):
        if q > 0:
            self.turnShow = self.handle.rightTurn(q)    
        elif q < 0:
            self.turnShow = self.handle.leftTurn(q)      
        elif q == 0:
            self.turnShow = "직진" 

#PohamCar Class 가져와서 사용하기
#Class.py
#상속
from pack.PohamCar import PohamCar

tom = PohamCar("tom")
tom.TurnHandle(20)
print(tom.ownerName+"의 회전량은 "+tom.turnShow+str(tom.handle.quantity))
tom.TurnHandle(-30)
print(tom.ownerName+"의 회전량은 "+tom.turnShow+str(tom.handle.quantity))
oscar = PohamCar("oscar")
oscar.TurnHandle(0)
print(oscar.ownerName+"의 회전량은 "+oscar.turnShow+str(oscar.handle.quantity))
'''
tom의 회전량은 우회전20
tom의 회전량은 좌회전-30
oscar의 회전량은 직진0
'''

```
<br>
###  메소드 오버로딩
오버로딩이란 상위클래스안에 선언된 메소드를 하위클래스에서 새로 선언 혹은 정의하는 것  
<a href="https://wjddyd66.github.io/java/2019/06/14/Inheritance.html">오버로딩 참조</a><br>

```python
#Method 오버로딩
class Animal():

    def __init__(self, name):
        self.name = name

    def walk(self):
        print('{} walk'.format(self.name))

    def eat(self):
        print('{} eat'.format(self.name))

    def greet(self):
        print('{} greet'.format(self.name))

class Dog(Animal):

    def __init__(self, name):
        self.name = name

    def bark(self):
        print('{} bark to you for greeting'.format(self.name))

    def greet(self): # Animal Method 오버로딩
        self.bark()


animal = Animal('my_animal') # Animal 인스턴스 생성
my_dog = Dog('Puppy') # Dog 인스턴스를 생성
animal.greet() # Animal 인스턴스의 greet 메소드를 호출
my_dog.greet() # Dog 인스턴스의 greet 메소드를 호출

animal.walk() # Animal 인스턴스의 walk 메소드를 호출
my_dog.walk() # Dog 인스턴스의 walk 메소드를 호출

'''
my_animal greet
Puppy bark to you for greeting
my_animal walk
Puppy walk
'''

```
<hr>
내용참조:<a href="https://andamiro25.tistory.com/50">Andamiro25 블로그</a><br>
내용참조:<a href="https://light-tree.tistory.com/95">잡탕찌개 블로그</a><br>
참조:<a href="https://github.com/wjddyd66/Python/tree/master/Class">원본코드</a><br>
코드에 문제가 있거나 궁금한 점이 있으면 wjddyd66@naver.com으로  Mail을 남겨주세요.