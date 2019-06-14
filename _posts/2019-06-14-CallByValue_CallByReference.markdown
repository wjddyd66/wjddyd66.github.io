---
layout: post
title:  "자바의 CallByValue, CallByReference"
date:   2019-06-14 10:45:20 +0700
categories: [JAVA]
---

### 변수의 저장 방식  
CallByValue, CallByReference의 설명의 들어가기 앞서 컴퓨터의 값의 저장방식에 대해서 알아야 한다. 밑의 그림을 보면 컴퓨터이 저장 방식을 알 수 있다.

![Screenshot broadcast](https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/_posts/Reference.png"Screenshot broadcast")

1. 주소 - 저장되어있는 공간
2. 행위 - 저장되어있는 공간에 가지고 있는 값

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
참조 사이트:<https://sleepyeyes.tistory.com/11>
원본코드: <https://github.com/wjddyd66/JAVA/blob/master/CallByValue_CallByReference.java>
코드에 문제가 있거나 궁금한 점이 있으면 wjddyd66@naver.com으로  Mail을 남겨주세요.


