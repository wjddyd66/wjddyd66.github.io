---
layout: post
title:  "JAVA-Polymorphism"
date:   2019-06-14 12:57:20 +0700
categories: [JAVA]
---

### 다형성 
다형성이란 같은 부모를 가진 다른 자식을 상황에 맞게 사용하는 것을 의미한다.  
<span style ="color: red">**하나의 Type(부모)로서 다른 결과를 얻을 수 있다. => 객체를 부품화하여 유지 보수를 용이하게 한다.**</span> 

<img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/_posts/Polymorphism.PNG" height="300" width="600" />

참조 :<a href="https://m.blog.naver.com/PostView.nhn?blogId=heartflow89&logNo=220979244668&proxyReferer=https%3A%2F%2Fwww.google.com%2F">JOKER 블로그</a>

밑의 코드를 보게 되면 Car라는 하나의 Interface을 상속받는 Sonata와 Genesis가 존재한다. Car가 Interface이므로 Sonata와 Genesis가 같은 Method를 Override하지만 Method의 내용은 다르므로 다른 결과를 얻을 수 있다.

{% highlight java %}
//Interface 정의
public interface Car {
	void name();
	void price();
}

//Grenger Class 정의
public class Grenger implements Car{
	//Car Method Override
	@Override
	public void name() {
		System.out.println("Grenger");
	}
	@Override
	public void price() {
		System.out.println("8000");	
	}
}

//Sonata Class 정의
public class Sonata implements Car{
	
	//Car Method Override
	@Override
	public void name() {
		System.out.println("Sonata");
	}
	@Override
	public void price() {
		System.out.println("4000");	
	}
}

//Polymorphism 예제
public class Polymorphism {
	public static void main(String[] args) {
		Car c1 = new Grenger();
		Car c2 = new Sonata();
		c1.name();
		c1.price();
		c2.name();
		c2.price();
		/*
		Car 라는 동일한 Type을 가지지만
		Car의 자식 중 다른 Sonata 와 Grenger를 사용하여
		Grenger 8000 Sonata 4000라는 다른 결과가 나온다
		 */
	}
}
{% endhighlight %}

<hr>
참조: <a href="https://github.com/wjddyd66/JAVA/tree/master/Polymorphism">원본코드</a>
코드에 문제가 있거나 궁금한 점이 있으면 wjddyd66@naver.com으로  Mail을 남겨주세요.