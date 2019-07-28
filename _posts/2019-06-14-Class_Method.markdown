---
layout: post
title:  "JAVA-클래스와 메소드"
date:   2019-06-14 09:45:20 +0700
categories: [JAVA]
---

### 자바의 클래스: 클래스는 객체를 만들기 위하여 정의  
여러가지의 공통된 객체를 만들때 공통된 것을 정의하기 위하여 사용한다.  
클래스는 선언만 되어있는 껍질 이다.  
선언되어 있는 클래스로서 객체를 만들 수 있다.  
(객체 = Object = 인스턴스)  
클래스 => 붕어빵 틀, 객체 => 붕어빵  
하나의 클래스는 속성과 행위로 이루어져 있다.(속성 or 행위 or 속성 + 행위)  
1. 속성 - 멤버 변수
2. 행위 - 메소드

### 자바의 메소드: Input과 Output + Logic이 결합된 하나의 Function  
한번의 Logic이 아닌 여러군대에서 사용되는 Function이 있을경우 불러다 계속해서 사용하기 위하여 정의  
Format : Access Modifer OutputType(Input type Input Value...)  
<span style ="color: red">**메소드 중 클래스의 초기 조건을 설정해 주는 메소드이다. 객체 생성 시 가장 먼저 수행, 생성자가 정의하지 않을 경우 컴파일러가 만들어 주게 된다. => 아무런 조건이 없는 상태로 만들어 줌**</span>
{% highlight java %}

	//Class_Method.java
	public class Class_Method {
	//private 변수 선언
	private int a=3;
	//private 변수에 접근하기 위하여 Getter public Method 선언
	public int getA() {
		return a;
	}
	
	//생성자로서 객체가 생성 될때 초기 조건을 주어줄 수 있다.
	public Class_Method() {
		System.out.println("생성자 생성 완료");
		this.a =4;
	}
	
	/*
	Method 선언으로서 
	public => Access Modifier
	int => return Type
	abc => Class 이름 
	int a => input Type
	*/
	public int abc(int a) {
		return a;
	}
	}
	
	//Object.java
	public class Object {
		public static void main(String[] args) {
			//앞에서 선언한 Object 선언
			Class_Method object = new Class_Method();
			System.out.println(object.abc(3));
			System.out.println(object.getA());
			/*3 4
			Object 가 생성되면서 Constructor 실행
			System.out.println("생성자 생성 완료");
			this.a =4;
			가 실행되므로 private 변수 a가 4로 값이 변하게 되었다.
			*/
		}
	}

{% endhighlight %}
<hr>
참조: <a href="https://github.com/wjddyd66/JAVA/tree/master/Class_Method">원본코드</a><br>
코드에 문제가 있거나 궁금한 점이 있으면 wjddyd66@naver.com으로  Mail을 남겨주세요.


