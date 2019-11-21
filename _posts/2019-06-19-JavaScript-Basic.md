---
layout: post
title:  "JavaScript-Basic"
date:   2019-06-19 11:00:00 +0700
categories: [Web]
---
<span style ="color: red">**본 글은 <a href="https://joshua1988.github.io/web-development/javascript/javascript-basic-summary">Captain Pangyo 블로그</a>을 보고 연습한 글입니다.**</span><br>

###  JavaScript
HTML과 CSS는 정적인 언어이다. 정적이라는 것은 브라우저를 통해서 화면에 출력하는 것이지 그 값을 변경시킬 수 없다는 의미이다.  
Web상에서 정적인 HTML과 CSS의 한계를 극복하기 위하여 사용하는 언어이다.  
자바 스크립트는 객체기반의 스크립트 언어이다.  
<link rel = "stylesheet" href ="/static/css/bootstrap.min.css">
<table class="table" style="width:100%">
	<tbody>
	<tr>
		<td>HTML</td><td>뼈대</td>
	</tr>
	<tr>
		<td>CSS</td><td>디자인</td>
	</tr>
		<tr>
		<td>JavaScript</td><td>웹의 동작</td>
		</tr>
	</tbody>
</table>
<br>
###  자바스크립트의 특징
1. 자바스크립트는 객체 기반의 스크립트 언어이다.
2. 자바스크립트는 동적이며, 타입을 명시할 필요가 없는 인터프리터 언어이다.
3. 자바스크립트는 객체 지향적 프로그래밍과 함수형 프로그래밍을 모두 표현 가능하다.  


### 자바스크립트 타입
기본 타입
1. Number - 실수
2. String - 문자열
3. Boolean - True, False
4. undefined - 변수에 값이 할당되지 않을 때 자동으로 할당되는 값
5. null - 개발자가 의도적으로 할당하는 값. typeof 값이 Object 로 반환. 따라서 === 로 확인

```js
var nullCheck = null;
console.log(typeof nullCheck === null);
console.log(nullCheck === null);
```
<br>
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/JavaScript/Js1.JPG" height="150" width="600" /></div>
<br>

참조 타입(객체 타입)
1. Object
2. Array - 배열도 객체로 취급
3. Function - 함수도 객체로 취급

Object 끼리는 값이 같아도 객체는 다르다고 판단된다. => 주소가 다르기 때문이다.  
Object의 주소가 같아야 같다고 판단된다.  
<span style ="color: red">**이러한 실제 위치를 가르키는 것을 포인터(Pointer)라고 한다.**</span>  
```js
var objA = { value : 20 };
var objB = { value : 20 };
console.log( objA == objB );
objA = objB;
console.log( objA == objB );
```
<br>
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/JavaScript/Js2.JPG" height="150" width="600" /></div>
<br>
### NaN (Not a Number)
수치 연산을 해서 정상적인 값을 얻지 못할 때 발생하는 에러  

```js
console.log(1 - 'hello'); // NaN
console.log(1 + 'hello'); // 1hello
```
<br>
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/JavaScript/Js3.JPG" height="150" width="600" /></div>
<br>

### delete 연산자
객체 프로퍼티를 삭제하는 기능. 객체 삭제는 불가능

```js
// 1. 객체 프로퍼티를 삭제
var foo = {
  name: 'foo',
  nickname: 'pangyo'
};

delete foo.nickname;
console.log(foo.nickname);
console.log(foo); // {name: "foo"}
// 2. delete 로 객체를 삭제할 경우 (변화 없음)
var foo = {
  name: 'foo',
  nickname: 'pangyo'
};

delete foo;
console.log(foo); // {name: "foo", nickname: "pangyo"}
```
<br>
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/JavaScript/Js4.JPG" height="150" width="600" /></div>
<br>

### 객체의 모든 연산은 참조 값을 처리
값 비교시에 사용하는 == 를 적용한 예제를 보자.

```js
var a = 10;
var b = 10;

var objA = {
  value: 100
};
var objB = {
  value: 100
};
var objC = objB;

console.log(a == b); // true
console.log(objA == objB); // false
console.log(objB == objC); // true
```
<br>
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/JavaScript/Js5.JPG" height="150" width="600" /></div>
<br>
## Array 랑 Object 구분 방법
<table class="table">
	<tbody>
	<tr>
		<td>Array</td><td>Object</td>
	</tr>
	<tr>
		<td>속성</td><td>요소</td>
	</tr>
	<tr>
		<td>키와 값</td><td>값</td>
	</tr>
	<tr>
		<td>순서로 구별</td><td>키 값으로 구별</td>
	</tr>
	</tbody>
</table>
<br>
```js
var arr = [];
var obj = {};

console.log(arr.constructor.name);
console.log(obj.constructor.name);
```
<br>
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/JavaScript/Js6.JPG" height="150" width="600" /></div>
<br>

## delete & splice 연산자 in 배열
배열에서 delete 를 사용하면 요소의 값만 undefined 로 변경하고, 해당 요소 index 를 지우지는 않는다.  
반대로 splice 는 해당 요소 전체를 아예 잘라내서 없앤다.  
웹 브라우저로 출력하였을때 undefined 가 empty로 나오게 된다.  
```js
var arr = [1, 2, 3];
delete arr[1];
console.log(arr); // [1, empty, 3]
var arr = [1, 2, 3];
arr.splice(1, 1);
console.log(arr); // [1, 3]
```
<br>
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/JavaScript/Js7.JPG" height="150" width="600" /></div>
<br>

## typeof 연산자
각 데이터 타입에 대한 typeof 수행결과는 다음과 같다.

```js
var num = 10;
var str = "a";
var boolean = true;
var obj = {};
var undefined;
var nullValue = null;
var arr = [];
function func() {};

console.log(typeof num); // number
console.log(typeof str); // string
console.log(typeof boolean); // boolean
console.log(typeof obj); // object
console.log(typeof undefined); // undefined
console.log(typeof nullValue); // object (null 은 object)
console.log(typeof arr); // object (배열도 object)
console.log(typeof func); // function
```
<br>
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/JavaScript/Js8.JPG" height="150" width="600" /></div>
<br>

## == 연산자와 === 연산자
== 와 === 의 가장 큰 차이점은 값 뿐만 아니라 타입까지 체크하느냐이다.
또한 == 는 수행시에 타입이 다를 경우 타입을 일치시켜 값을 비교하는 특징이 있다.

```js
console.log(1 == '1'); // true
console.log(1 === '1'); // false
```
<br>
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/JavaScript/Js9.JPG" height="150" width="600" /></div>
<br>

## 함수 호이스팅
함수 호이스팅이란 함수선언문으로 생성된 함수가 해당 소스의 유효범위의 맨 위로 끌어올려진다.  
함수 호이스팅이란 var "function_name"으로서 함수를 부르기 전 위에 선언하는 것 이다.  
함수 호이스팅 전:  
```js
add(2, 3); // add is not a function
var add = function (a, b) {
  return a + b;
};
add(4, 5);
```
<br>
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/JavaScript/Js10.JPG" height="150" width="600" /></div>
<br>
함수 호이스팅 하였지만, 개인적인 환경에서는 같은 Error가 발생하게 되었다.  
```js
var add;
add = function (a, b) {
  return a + b;
};
console.log(add(4, 5))
console.log(add(2, 3))
```
<br>
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/JavaScript/Js12.JPG" height="150" width="600" /></div>
<br>
함수 선언 후 수행:  
```js
var add;
add = function (a, b) {
  return a + b;
};
console.log(add(4, 5))
console.log(add(2, 3))
```
<br>
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/JavaScript/Js11.JPG" height="150" width="600" /></div>
<br>

## 함수의 length 속성
함수의 length란 인수(argument)의 개수이다. 
```js
function func1(a) { return a; }
function func2(a, b) { return a + b; }
function func3(a, b, c) { return a + b + c; }
function func4(a) { return a + a + a; }

console.log('func1 length : ' + func1.length); // func1 length : 1
console.log('func2 length : ' + func2.length); // func2 length : 2
console.log('func3 length : ' + func3.length); // func3 length : 3
console.log('func4 length : ' + func4.length); // func4 length : 1
```
<br>
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/JavaScript/Js13.JPG" height="150" width="600" /></div>
<br>

## 내부 함수
함수 내부에 선언한 함수이다.  
하나의 외부 함수에 여러개의 내부 함수가 존재할 수 있다.  



```js
function parent() {
  var a = 10;
  var b = 20;

  function child1() {
    var b = 30;
    console.log(a);
    console.log(b);
  }
  function child2() {
    var b = 40;
    console.log(a);
    console.log(b);
  }
  child1();
  child2();
}
parent(); // 10, 30, 10, 40
child1(); // child is not defined
```
<br>
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/JavaScript/Js14.JPG" height="150" width="600" /></div>
<br>

## 생성자 함수
일반 객체 선언과 다르게 여러 개의 객체를 찍어낼 수 있는 함수.
함수명 맨 앞 글자는 대문자, 호출 시에 new 사용.

```js
function Developer(name, stack, city) {
  this.name = name;
  this.stack = stack;
  this.city = city;
}
var dev = new Developer('captain', 'web', 'pangyo');
var devops = new Developer('hulk', 'devops', 'seoul');
console.log(dev); // Developer {name: "captain", stack: "web", city: "pangyo"}
console.log(devops); // Developer {name: "hulk", stack: "devops", city: "seoul"}
```
<br>
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/JavaScript/Js15.JPG" height="150" width="600" /></div>
<br>

## instaceof
instanceof란 생성자의 prototype 속성이 객체의 프로토타입 어딘가 존재하는지 판별합니다.  
```js
function Car(make, model, year) {
  this.make = make;
  this.model = model;
  this.year = year;
}
var auto = new Car('Honda', 'Accord', 1998);
var auto2 = new Car('Honda', 'Accord', 1998);
var auto3 = Car('Honda', 'Accord', 1998);

console.log(auto instanceof Car); //true
console.log(auto2 instanceof Car); //true
console.log(auto3 instanceof Car); //false
```
<br>
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/JavaScript/Js16.JPG" height="150" width="600" /></div>
<br>

## instaceof 를 활용한 생성자 함수 구분법
자바스크립트는 생성자 함수 형식이 별도로 없기에 기존 함수에 new만 붙여주면 생성자 함수 생성이 가능하다.
따라서, 생성자 함수가 아닌데 new 를 붙이는 경우를 대비해서 아래와 같은 기법을 적용할 수 있다.
**대부분의 오픈소스 라이브러리에서 사용하는 패턴**

```js
function Func(arg) {
  // instanceof 로 생성자 함수임을 확인
  if (!(this instanceof arguments.callee)) // 'this instanceof 함수명' 도 가능
    return new Func(arg);
  this.value = arg || 0;
}

var a = new Func(100);
var b = Func(200);
console.log(a.value);
console.log(b.value);
```
<br>
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/JavaScript/Js17.JPG" height="150" width="600" /></div>
<br>

## 프로토타입(Prototype)
자바스크립트는 객체지향 언어이다.  
하지만 자바 스크립트에서는 클래스라는 개념이 없다.  
따라서 상속의 기능이 없기 때문에 프로토타입을 기반으로 상속을 흉내내도록 구현해 사용한다.  
상속을 받음으로 인하여 변수가 할당되는 메모리의 크기를 크게 줄일 수 있는 장점이 생기게 된다.  
<span style ="color: red">**프로토타입 체이닝: 해당 함수에 존재하지 않는 속성, 메서드를 부모 객체(Prototype 객체)를 찾음**</span><br>
참고로, 자바스크립트 자바와 마찬가지로 모든 객체의 최상위 부모 객체는 Object 객체다.  
```js
function Person() {}
Person.prototype.eyes = 2;
Person.prototype.nose = 1;
var kim  = new Person();
var park = new Person();
console.log(kim.eyes); //Kim에 eyes라는 property가 존재하지 않으나 부모 객체의 값을 가져와서 출력하는 것을 볼 수 있다.
console.log(park.eyes);// => 2
```
<br>
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/JavaScript/Js18.JPG" height="150" width="600" /></div>
<br>

## 프로토타입 객체 메서드 재정의
부모에서 정의된 메서드를 자식이 Override하여 재정의 할 수 있다.  

```js
// 부모 클래스 
function MyParent() { 
	this.property1 = 'data1'; 
	console.log('MyParent'); 
} 
MyParent.prototype.method1 = function () { 
console.log('property1 = ' + this.property1); 
}; 
// 자식 클래스 
function MyChild() { 
console.log('MyChild'); } 
// 부모 클래스 상속하기 MyChild.prototype = new MyParent(); 
// 생성자 설정 MyChild.prototype.constructor = MyChild; 
/* 
------------------ * 메서드 오버라이드 * ------------------ 
*/ 
MyChild.prototype.method1 = function () { 
console.log('프로퍼티 1은 = ' + this.property1 + ' 입니다.'); 
}; 
// 자식 인스턴스 생성 
var child = new MyChild(); 
// 메서드 호출 
child.method1(); 
```
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/JavaScript/Js19.JPG" height="150" width="600" /></div>
<br>
## 즉시 실행 함수
함수를 정의함과 동시에 바로 실행하는 함수. 함수를 다시 호출할 수 없다는 특징이 있다.
따라서, 최초 한 번의 실행만 요구되는 초기화 코드에 적합하다.
jQuery 와 같은 오픈소스 라이브러리들의 구조.  
(로 묶고 바로 Argument를 대입)

```js
/*
Format
function [name]([param1[, param2[, ..., paramN]]]) {
   statements
}
*/
(function (name) {
  console.log('This is the immediate function : ' + name);
})(['foo',['foo2']]);
```
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/JavaScript/Js20.JPG" height="150" width="600" /></div>
<br>
## 클로져
클로져는 독립적인 변수를 가르키는 함수이다. 또는 클로저 안에 정의된 함수는 만들어진 환경을 기억한다.  
클로져가 변수에 접근할 때에는 변수명 앞에 underscore(_)를 써서 접근가능하도록 하였다.  

```js
function Hello(name) {
  this._name = name;
}

Hello.prototype.say = function() {
  console.log('Hello, ' + this._name);
}

var hello1 = new Hello('승민');
var hello2 = new Hello('현섭');
var hello3 = new Hello('유근');

hello1.say(); // 'Hello, 승민'
hello2.say(); // 'Hello, 현섭'
hello3.say(); // 'Hello, 유근'
hello1.name = 'anonymous';
hello1.say(); // 'Hello, anonymous'
hello1._name = 'anonymous';
hello1.say(); // 'Hello, anonymous'
```
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/JavaScript/Js21.JPG" height="150" width="600" /></div>
<br>

## 클로져를 통한 은닉화
클로져가 변수에 접근하기 위해서는 변수명 앞에 underscore(_)를 써서 접근 가능하다.  
이러한 접근이 싫으면 미리 _변수명 = Value로서 선언함으로 인하여 은닉화를 구현할 수 있다.  

```js
function Hello(name) {
  this._name = name;
}

Hello.prototype.say = function() {
  console.log('Hello, ' + this._name);
}

var hello1 = new Hello('승민');
var hello2 = new Hello('현섭');
var hello3 = new Hello('유근');

hello1.say(); // 'Hello, 승민'
hello2.say(); // 'Hello, 현섭'
hello3.say(); // 'Hello, 유근'
hello1.name = 'anonymous';
hello1.say(); // 'Hello, anonymous'
hello1._name = 'anonymous';
hello1.say(); // 'Hello, anonymous'
```
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/JavaScript/Js22.JPG" height="150" width="600" /></div>
<br>

## 실행 컨텍스트를 이해하기 위한 자바스크립트 동작과정
1. Variable Object: 실행에 필요한 여러 정보들을 담을 객체 생성
 - 변수
 - Parameter, Argument
 - 함수 선언
2. Scope Chain: 해당 전역 또는 함수가 참조할 수 있는 변수, 함수선언 등의 정보를 담고있는 전역객체 또는 활성객체의 리스트
3. this: this 프로퍼티에는 this값이 할당된다. this에 할당되는 값은 함수 호출 패턴에 의해 결정된다.


비동기 실행 방식인 setTimeout 를 이용한 예제이다.  
setTimeout 이 지연시간이 0 이라고 할지라도 실행 컨텍스트가 다르기 때문에 1,4 가 먼저 출력된다.
```js
console.log("1");
function exec() {
  setTimeout(function() {
    console.log("2");
  }, 3000);
  setTimeout(function() {
    console.log("3");
  }, 0);
  console.log("4");
  setTimeout(function() {
    console.log(5);
  }, 1000);
}
console.log(exec());
// 위 코드 실행 결과 : 1, 4, 3, 5, 2
```
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/JavaScript/Js23.JPG" height="150" width="600" /></div>
<br>

이번엔 for 문과 setTimeout 이다.  
실행되는 메인 컨텍스트와 setTimeout 이 실행되는 컨텍스트가 다르기 때문에
일반 프로그래밍 지식 관점에서는 0,1,2,3,4 이라고 추측하겠지만, 실제로는 for 문의 실행이 모두 끝난 후에
setTimeout 의 콜백 함수가 실행되기 때문에 숫자 5가 다섯 번 출력된다.  

```js
var i;
for (i = 0; i < 5; i++) {
  setTimeout(function() {
    console.log(i); // 5, 5, 5, 5, 5
  }, 1000);
}
```
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/JavaScript/Js24.JPG" height="150" width="600" /></div>
<br>

## arguments 객체
함수 호출시에 넘겨진 실제 인자 값을 가진 배열  
arguments 의 활용 : 메서드에 넘겨 받을 인자의 개수를 모를 때 유용
```js
function sum() {
  for (var i = 0, result = 0; i < arguments.length; i++) {
    result += arguments[i];
  }
  return result;
}
console.log(sum(1,2,3)); // 6
console.log(sum(1,2,3,4,5,6)); // 21
```
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/JavaScript/Js25.JPG" height="150" width="600" /></div>
<br>

## apply() & call()
다른 객체 대신 메소드를 호출하는데 사용된다.  
이 메서드를 사용하여 함수의 this 객체를 원래 컨텍스트에서 thisObj로 지정된 새 객체로 변경할 수 있다.  
<table class="table">
	<tbody>
	<tr>
		<td>종류</td><td>구문</td><td>매게변수</td>
	</tr>

	<tr>
		<td>call</td>
		<td>fun.call([thisObj[, arg[, arg2[, ...]]]</td>
		<td>
			<ul>
				<li>fun: 가져다 쓸 메소드</li>
				<li>thisObj(선택 사항): 현재 객체로 사용될 객체</li>
				<li>arg1,arg2,argN(선택 사항): 메소드에 전달될 인수 목록</li>
			</ul>
		</td>
	</tr>
	
		<tr>
		<td>apply</td>
		<td>fun.apply([thisObj[,argArray]])</td>
		<td>
			<ul>
				<li>fun: 가져다 쓸 메소드</li>
				<li>thisObj(선택 사항): 현재 객체로 사용될 객체</li>
				<li>argArray: 메소드에 전달될 인수 집합</li>
			</ul>
		</td>
	</tr>
	
	</tbody>
</table>
<br>

```js
function sum() {
  var args1 = Array.apply(arguments);
  args1.push(100); // 0: 100
  console.dir(args1); // Array(1)

  var args2 = Array.prototype.slice.apply(arguments);
  args2.push(100); // 3: 100
  console.dir(args2); // Array(4)
}
sum(1,2,3);
```
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/JavaScript/Js25.JPG" height="150" width="600" /></div>
<br>

```js
function user(firstName, lastName, age) {
    this.firstName = firstName;
    this.lastName = lastName;
}
user.apply(window, ['pangyo', 'captain']); // user.call(window, 'John', 'Doe'); 와 같음

console.log(window.firstName); // 'pangyo'
console.log(window.lastName); // 'captain'
```
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/JavaScript/Js27.JPG" height="150" width="600" /></div>
<br>

## this 바인딩
일반적으로 함수 내부에서 this를 사용하면 전역 스코프(window)에 접근한다.  
<span style ="color: red">**하지만 현재 작업하고 있는 환경에서는 this가 적용되지 않았다.  Online Complier 환경에서 Complie하고 있어서 생기는 문제인 것 같다.**</span><br>


```js
// 함수 선언식
var text = 'global';
function binding() {
  var text = 'local';
  console.log(this.text); // 'global'
  console.log(this); // Window {stop: ƒ, open: ƒ, alert: ƒ, confirm: ƒ, prompt: ƒ, …}
}
binding();
```
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/JavaScript/Js28.JPG" height="150" width="600" /></div>
<br>
객체의 속성에서 함수를 선언하고 this를 사용하면 해당 객체를 접근한다.

```js
var text = 'global';
var binding = {
  text: 'local',
  printText: function () {
    console.log(this.text); // 'local'
    console.log(this); // {text: "local", printText: ƒ}
  }
};
binding.printText();
```

## 스코프 체인을 이해하기 위한 예제
스코프 체인이란 실행 시점에서 identifiers(식별자)를 찾는 것이다.  
아래 예제는 지역변수와 전역 변수의 스코프이다.  
지역변수가 우선시 되는것을 확인할 수 있다.  

```js
// ex.1
var a = 1;
var b = 2;
function func() {
  var a = 10;
  var b = 20;
  console.log(a); // 10
  console.log(b); // 20
}
func();
console.log(a); // 1
console.log(b); // 2
```
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/JavaScript/Js29.JPG" height="150" width="600" /></div>
<br>
아래는 내부함수 innerfunc 에서 외부함수인 func 의 변수에 접근하고 있다.

```js
// ex.2
var a = 1;
function func() {
  var a = 2;
  function innerfunc() {
    return a;
  }
  console.log(innerfunc()); // 2
}
func();
```
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/JavaScript/Js30.JPG" height="150" width="600" /></div>
<br>
아래 예제는 func1 의 실행 컨텍스트가 전역이라는 것에 주목한다.

```js
// ex.3
var a = 1;
function func1() {
  return a;
}
function func2(func1) {
  var a = 2;
  console.log(func1()); // 1
}
func2(func1);
```
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/JavaScript/Js31.JPG" height="150" width="600" /></div>
<br>

<hr>
내용 참조: <a href="https://joshua1988.github.io/web-development/javascript/javascript-basic-summary">Captain Pangayo 블로그</a><br>
내용 참조: <a href="http://tcpschool.com/javascript/js_intro_basic">TCP School</a><br>
내용 참조: <a href="https://jsfiddle.net/z48w2s6o/35">Jsfiddle Code</a><br>
내용 참조: <a href="https://hyunseob.github.io/2016/08/30/javascript-closure">DailyEngineering 블로그</a><br>
내용 참조:<a href="https://beomy.tistory.com/4">버미노트 블로그</a><br>
코드에 문제가 있거나 궁금한 점이 있으면 wjddyd66@naver.com으로  Mail을 남겨주세요.