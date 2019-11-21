---
layout: post
title:  "Python-Type"
date:   2019-07-01 06:00:00 +0700
categories: [Python]
---

###  Python Type
<link rel = "stylesheet" href ="/static/css/bootstrap.min.css">
<table class="table">
	<tbody>
	<tr>
		<td>타입</td><td>순서</td><td>특징</td>
	</tr>
	<tr>
		<td>List</td><td>O</td><td>값 변경 가능</td>
	</tr>
		<tr>
		<td>Tuple</td><td>O</td><td>값 변경 불가능(List보다 빠르다)</td>
	</tr>
		<tr>
		<td>Set</td><td>X</td><td>중복 불가</td>
	</tr>
			<tr>
		<td>Dict</td><td>X</td><td>{key:value}형식</td>
	</tr>

	</tbody>
</table>
<br>

###  List
여러 요소들을 갖는 집합으로 새로운요소를 추가하거나 갱신, 삭제 가능.  
동적배열로서 자유롭게 확장 가능 및 요소 값 변경 가능  
순서가 있으므로 Index로거 접근 가능  
<span style ="color: red">**List Comprehension: 리스트의 [...] 괄호 안에 for 루프를 사용하여 반복적으로 표현식(expression)을 실행해서 리스트 요소들을 정의하는 특별한 용법**</span><br>

```python
#List
#List 선언
a=[1,2,3]
b=[10, a, 10.5, True, '문자열']

#List Indexing
print(a[0], a[1]) #1 2

#List Slicing
print(b[0:2], b[2], b[1][1], b[1][:3]) #[10, [1, 2, 3]] 10.5 2 [1, 2, 3]

#List 추가/삭제
family=["엄마","아빠","나"]
family.append("여동생") #추가
family.insert(0, "할아버지") #삽입
family.extend(["남동생", "이모", "고모"])
family+=["조카", "삼촌"]
family.remove("여동생") #삭제
family[1] = '엄마2' #Index로 접근 후 수정
del family[0]
print(family, " ", len(family)) #['엄마2', '아빠', '나', '남동생', '이모', '고모', '조카', '삼촌']   8

#List 검색
mylist = "This is a book That is a pencil".split()
i = mylist.index('book')  # i = 3 몇번째에 위치하는지
n = mylist.count('is')    # n = 2 is 가 몇개인지
print(i, n)

'''
List Comprehension
리스트의 [...] 괄호 안에 for 루프를 사용하여 반복적으로 표현식(expression)을 
실행해서 리스트 요소들을 정의하는 특별한 용법이 있는데, 이를 List Comprehension 이라 부른다. 
'''
list = [n ** 2 for n in range(10) if n % 3 == 0]
print(list)
'''
출력: [0, 9, 36, 81]
0부터 9까지 숫자들중 3으로 나눈 나머지가 0인 숫자에 대해 그 제곱에 대한 리스트를 구한 예이다.
'''

```
<br>

###  Tuple
여러 요소들을 갖는 집합으이나 새로운요소를 추가하거나 갱신, 삭제 <span style ="color: red">불가능. 즉, 한번 결정된 요소를 변경할 수 없는 Immutable 데이터 타입</span>  
항상 고정된 요소값을 갖기를 원하거나 변경되지 말아야 하는 경우 사용  
각 변수에 다른 변수를 할당할 수 있다.  

<br>

```python
#Tuple
#Tuple 선언
t=("a","b","c","d")
t="a","b","c","d"

#Tuple Indexing
print(t[1]) #b

#Tuple Slicing
print(t[0:3])#('a', 'b', 'c')

#Tuple 병합과 반복
# 병합
a = (1, 2)
b = (3, 4, 5)
c = a + b
print(c)   # (1, 2, 3, 4, 5)
 
# 반복
d = a * 3  # 혹은 "d = 3 * a" 도 동일
print(d)   # (1, 2, 1, 2, 1, 2)

#Tuple 변수 할장(각 요소를 다른 변수에 할당 가능)
name = ("John", "Kim")
print(name)
# 출력: ('John', 'Kim')
 
firstname, lastname = ("John", "Kim")
print(lastname, ",", firstname)
# 출력: Kim, John

```
<br>

###  Set
<span style ="color: red">**중복이 없는 요소들로만 구성된 집합 컬랙션**</span><br>
내부적으로 순서가 없기 때문에 Indexing 불가.  


```python
#Set
#Set 선언
a={1,2,2,3}
print(a, len(a)) #{1, 2, 3} 3

'''
Set Method
Union(): 두 튜플을 중복없이 합침
intersection(): 두 튜플 사이에 공통된 값을 추출
'''
b={3,4}
print(a.union(b)) #{1, 2, 3, 4}
print(a.intersection(b)) #{3}
print(a-b, a|b, a&b) #{1, 2} {1, 2, 3, 4} {3}

'''
Set 추가 삭제
add(): 요소를 추가
update(): 값을 업데이트, 기존의 값이 있으면 그냥 덮어쓴다.
discard(): 요소값을 삭제(요소값이 있으면 삭제하고, 없으면 그냥 무시한다.)
remove(): 요소값을 삭제(요소값이 없는데 삭제를 시도하면 에러가 발생)
'''
a.add(4) 
print(a) #{1, 2, 3, 4}

a.update({2,5,6}) #set
a.update((7,8)) #tuple
a.update([9,10]) #list
print(a) #{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

a.discard(2) 
a.remove(3)
print(a) #{1, 4, 5, 6, 7, 8, 9, 10}

```
<br>

###  Dict
<span style ="color: red">**{Key: Value}쌍을 요소로 갖는 컬랙션. Map의 형식과 같다.  
Hash Table구조로 이루워져 있으므로 Key로 신속하게 Value를 찾을 수 있다.**</span><br>  
keys(): dict_key객체를 리턴, values: dict_values객체를 리턴  

```python
#Dict
#Dict선언
mydic=dict(k1=1, k2="kbs", k3=1.2)

#Dict 출력
dic={"파이썬":"뱀", "자바":"커피", "스프링":"용수철"}
print(dic) #{'파이썬': '뱀', '자바': '커피', '스프링': '용수철'}
#Dict는 순서가 없기 때문에 입력한 순서대로 출력되는 것은 아니다.
print(len(dic)) #3
print(dic['자바']) #커피

#Dict Method
print(dic.keys()) #dict_keys(['파이썬', '자바', '스프링'])
print(dic.values()) #dict_values(['뱀', '커피', '용수철'])

#Dict 추가,삭제
dic['오라클']="디비" 
print(dic) #{'파이썬': '뱀', '자바': '커피', '스프링': '용수철', '오라클': '디비'}
del dic["오라클"] #요소삭제
print(dic) #{'파이썬': '뱀', '자바': '커피', '스프링': '용수철'}
dic.clear() #전부 삭제
print(dic) #{}

```
<br>
<hr>
내용참조:<a href="https://blog.naver.com/PostView.nhn?blogId=mint3081&logNo=221533482749&parentCategoryNo=&categoryNo=49&viewDate=&isShowPopularPosts=false&from=postList">천프로 블로그</a><br>
내용참조:<a href="http://pythonstudy.xyz/python/article/13-%EC%BB%AC%EB%A0%89%EC%85%98--Tuple">예제로 배우는 Python 프로그래밍</a><br>
내용참조:<a href="http://pythonstudy.xyz/python/article/14-%EC%BB%AC%EB%A0%89%EC%85%98--Dictionary">예제로 배우는 Python 프로그래밍</a><br>
내용참조:<a href="http://pythonstudy.xyz/python/article/13-%EC%BB%AC%EB%A0%89%EC%85%98--Tuple">예제로 배우는 Python 프로그래밍</a><br>
참조:<a href="https://github.com/wjddyd66/Python/tree/master/Type">원본코드</a><br>
코드에 문제가 있거나 궁금한 점이 있으면 wjddyd66@naver.com으로  Mail을 남겨주세요.