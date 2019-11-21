---
layout: post
title:  "SingleTon"
date:   2019-06-14 12:51:20 +0700
categories: [others]
---

### SingleTon: 여러명이 공유하는 웹 환경에서 하나만 사용하는 디자인 패턴.
<br><br>

{% highlight java %}
public class SingletonTest {
	int kor=100;
	
	// gof의 디자인패턴 중 싱글톤 패턴
	private static SingletonTest SingletonTest = new SingletonTest();
	public static SingletonTest getInstance() {
		return SingletonTest;
	}
	
	public void abc() {
		System.out.println("abc method");
	}
}

public class SingletonMain {
	public static void main(String[] args) {
		SingletonTest test1=new SingletonTest();
		SingletonTest test2=new SingletonTest();
		System.out.println(test1);
		System.out.println(test2);

		System.out.println("-");
		SingletonTest ex1=SingletonTest.getInstance();
		SingletonTest ex2=SingletonTest.getInstance();
		SingletonTest ex3=SingletonTest.getInstance();
		System.out.println(ex1+" "+ex2+" "+ex3);
		// 객체를 3개 생성했지만 셋의 주소는 결국 모두 같다. 
		System.out.println(ex1.kor);
		ex1.abc();
		
		System.out.println("-");
		Calendar cal=Calendar.getInstance();
		// new 안하고 객체를 생성하고 있다 -> 싱글톤 패턴 사용의 예
		System.out.println(Calendar.YEAR);
		int year=cal.get(Calendar.YEAR);
		System.out.println("연도는 "+year);
	}
}
{% endhighlight %}
<br>

<hr>
코드에 문제가 있거나 궁금한 점이 있으면 wjddyd66@naver.com으로  Mail을 남겨주세요.

