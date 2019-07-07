---
layout: post
title:  "Python-Web Scrapping,Crawling"
date:   2019-07-06 09:40:00 +0700
categories: [Python]
---

###  Web Scrapping,Crawling
1. Web Scrapping: 웹의 데이터를 일부만 가져오는 작업
2. Web Crawling: 웹사이트에 주기적으로 방문하여 데이터를 가져오는 작업

<span style ="color: red">**XML, Json, HTML**</span>을 Web Scrapping,Crawling을 통하여 가져오고 가공하는 작업을 할 것이다.  
3개의 Type은 DOM형식이므로 파일을 읽어오거나 ElementTree로서 가져올 수 있다.  
<a href="https://wjddyd66.github.io/web/2019/06/20/JavaScript-DOM,JQuery,Ajax.html">DOM 자세한 내용</a>

###  XML 자료 읽기
Local에 있는 my.xml자료를 읽는 과정이다.  
1. String Type으로 파일 읽어오기
2. ElementTree로서 가져오기

```python
#scrap1.py
# XML 자료처리
import xml.etree.ElementTree as et

# 방법1. 파일읽기 -> string 타입으로 가져온다.
xml_f = open("my.xml", mode="r", encoding="utf-8").read()
print(xml_f)
'''
<?xml version="1.0" encoding="UTF-8"?>
<items>
    <item>
        <name id="ks1">홍길동</name>
        <tel>010-111-1111</tel>
        <exam kor="100" eng="90" />
    </item>
    <item>
        <name id="ks2">고길동</name>
        <tel>010-111-2222</tel>
        <exam kor="88" eng="92" />
    </item>
</items>
'''
print(type(xml_f))#<class 'str'>

root = et.fromstring(xml_f) #str -> ElementTree 객체로 변환한다.
# 이렇게 변환하고 나면 ElementTree가 가지고 있는 명령어를 사용할 수 있다.
print(type(root))#<class 'xml.etree.ElementTree.Element'>
print(root.tag)# items
print(len(root)) #items는 2개의 자식을 가지고 있다.
print("*"*50)

# 방법2. ElementTree 객체로 직접 파싱하기 -> XML이 직접 온다.
xmlfile = et.parse("my.xml")
print(type(xmlfile))#<class 'xml.etree.ElementTree.ElementTree'>

root = xmlfile.getroot()
print(root.tag) #루트 태그를 반환한다. -> items
print(root[0].tag) #루트 태그의 0번째 자식을 반환한다. -> item
print(root[0][0].tag) 
#루트 태그의 0번째 자식의 0번째 자식을 반환한다. -> name
print(root[0][1].tag) # -> tel
print(root[0][0].attrib)  # {'id': 'ks1'}
print(root[0][2].attrib) # {'kor': '100', 'eng': '90'}

print(root[0][2].attrib.keys()) #dict_keys(['kor', 'eng'])
print(root[0][2].attrib.values()) #dict_values(['100', '90'])

print(root[0][2].attrib.get("kor")) #100

imsi = list(root[0][2].attrib.values())
print(imsi[0]+", "+imsi[1]) #100, 90

print("*"*50)
myname = root.find("item").find("name").text
#find() 를 이용해 자식의 요소명을 입력해주면 된다.
mytel = root.find("item").find("tel").text
print(myname+", "+mytel)
#홍길동, 010-111-1111

print("\n ▷ 반복처리하기---")
for child in root:
    print(child.tag)
    for child2 in child:
        print(child2.tag, child2.attrib)
'''
 ▷ 반복처리하기---
item
name {'id': 'ks1'}
tel {}
exam {'kor': '100', 'eng': '90'}
item
name {'id': 'ks2'}
tel {}
exam {'kor': '88', 'eng': '92'}
'''       
print("▷ 특정 요소의 속성값 얻기---")
for a in root.iter("exam"):
    print(a.attrib)
'''
▷ 특정 요소의 속성값 얻기---
{'kor': '100', 'eng': '90'}
{'kor': '88', 'eng': '92'}
'''   
print()
children = root.findall("item") #root 밑의 아이템을 전부 찾는다.
# find(), findall() 둘 다 있으니 적절히 활용할 것.
for chi in children:
    re_id =  chi.find("name").get("id")
    re_name =  chi.find("name").text
    re_tel =  chi.find("tel").text
    print(re_id, re_name, re_tel)
'''
ks1 홍길동 010-111-1111
ks2 고길동 010-111-2222
'''
```
<br>
###  XML 기상날씨 Scrapping
Web상에 존재하는 <a href="http://www.kma.go.kr/XML/weather/sfc_web_map.xml">XML파일</a>을 Local File(ftest.xml)로 저장 한뒤 Fil의 내용을 읽는 과정이다.  

```python
#scrap2.py
# 기상청 날씨정보 스크래핑 
import urllib.request
import xml.etree.ElementTree as et

try:
    webdata = urllib.request.urlopen("http://www.kma.go.kr/XML/weather/sfc_web_map.xml")
    #print(webdata)
    webxml = webdata.read() #binary 데이터로 읽어온다.
    webxml = webxml.strip().decode()
    # 바이너리를 문자열로 변환하는 작업이 필요하다.
    # 정해져있는 틀이기 때문에 항상 이런 식의 작업을 반복하게 될 것이다.
    #print(webxml)
    webdata.close()
    
    with open("ftest.xml", mode="w", encoding="utf-8") as f:
        f.write(webxml)
    
except Exception as e:
    print("err: ", e)
    
print("읽기 성공")

xmlfile = et.parse("ftest.xml")    
root = xmlfile.getroot()
print(root.tag)
print(root[0].tag)

children = root.findall("{current}weather")
print(children)

for i in children:
    y = i.get("year")
    m = i.get("month")
    d = i.get("day")
    h = i.get("hour")
    print(str(y)+"년 "+str(m)+"월"+str(d)+"일"+str(h)+"시 현재")
    
datas = []
for child in root:
    print(child.tag)
    for i in child:
        #print(i.tag)
        local_name =i.text
        re_ta = i.get("ta")
        re_desc = i.get("desc")
        datas+=[[local_name, re_ta, re_desc]]
        print(local_name+", 온도: "+str(re_ta)+" "+re_desc) 
print("건 수: ", len(datas))

print("*"*50)
'''
읽기 성공
{current}current
{current}weather
[<Element '{current}weather' at 0x000002837D3EE6D8>]
2019년 07월06일18시 현재
{current}weather
속초, 온도: 24.0 구름많음
북춘천, 온도: 30.0 맑음

...

산청, 온도: 29.8 구름조금
거제, 온도: 25.8 맑음
남해, 온도: 28.6 구름조금
건 수:  96
**************************************************
'''

```
<br>

###  BeautifulSoup
BeautifulSoup은 HTML,XML document 안에 있는 수많은 HTML 태그들을 사용하기 편한 Python객체 형태로 만들어준다.  
<span style ="color: red">** Json**</span>형태는 지원하지 않는다.  

```code
pip install requests
pip install bs4
pip install lxml
```
<br>
위의 코드를 cmd 창에 실행시키므로서 BeautifulSoup를 사용할 수 있다.  
아래 코드는 BeautifulSoup을 이용하여 naver안에있는 a링크를 읽어오는 code이다.  

```python
#scrap4.py
# 뷰티플숲으로 크롤링하기 -> 파이썬에서 가장 많이 사용하는 방법

import requests
from bs4 import BeautifulSoup

def go():
    base_url = "http://www.naver.com/index.html"

    #storing all the information including headers in the variable source code
    source_code = requests.get(base_url)
    print(source_code)

    #sort source code and store only the plaintext
    plain_text = source_code.text
    #print(plain_text)

    #converting plain_text to Beautiful Soup object so the library can sort thru it
    convert_data = BeautifulSoup(plain_text, 'lxml')
    print(type(convert_data)) #BeautifulSoup 객체가 생성된 것을 확인할 수 있다.

    #sorting useful information
    #for link in convert_data.findAll('a', {'class': 'h_notice'}):
    for link in convert_data.findAll('a'): # "a" 태그가 걸려있는 요소들을 전부 읽어들인다.
        href = base_url + link.get('href')  #Building a clickable url
        print(href)                          #displaying href
        
go()

'''
<Response [200]>
<class 'bs4.BeautifulSoup'>
http://www.naver.com/index.html#news_cast
http://www.naver.com/index.html#themecast
http://www.naver.com/index.html#time_square

...

http://www.naver.com/index.html/policy/spamcheck.html
http://www.naver.com/index.htmlhttps://help.naver.com/
http://www.naver.com/index.htmlhttps://www.navercorp.com/
'''
```
<br>
###  BeautifulSoup Method
1. find(): 해당 조건에 맞는 하나의 태그를 가져온다. 중복이면 가장 첫 번째 태그를 가져온다.
2. find_all(): 해당 조건에 맞는 모든 태그들을 가져온다.
3. prettify(): Html모양처럼 보기에 편하게 만들어주는 함수

<span style ="color: red">** 추가 요소**</span><br>
1. 정규표현식: 정규표현식을 활용하여 원하는 정보를 Filterring 하여 가져올 수 있다.
 - <a href="https://wjddyd66.github.io/others/2019/06/16/RegularExpression.html">정규표현식 자세한 내용</a>

2. CSS Selector를 활용하여 원하는 정보 Filterring

```python
#scrap5.py
from bs4 import BeautifulSoup

html_data = """
<html>
<body>
<h1>제목 태그</h1>
<p>뷰티플숲으로 읽기</p>
<p>원하는 자료 추출</p>
</body>
</html>
"""
print(type(html_data)) #<class 'str'>
soup = BeautifulSoup(html_data, "html.parser")
print(type(soup)) # <class 'bs4.BeautifulSoup'>
print()

h1 = soup.html.body.h1
print("h1: ", h1.string) #h1:  제목 태그
p1 = soup.html.body.p
print("p1: ", p1.string) #p1:  뷰티플숲으로 읽기
# 최초의 p 태그를 가져온다.
p2 = p1.next_sibling.next_sibling
print("p2: ", p2.string) # p2:  원하는 자료 추출

#
print("\n ▷ find() 메소드 사용하기 -----")
html_data2 = """
<html>
<body>
<h1 id="title">제목 태그</h1>
<p>뷰티플숲으로 읽기</p>
<p attr="my">원하는 자료 추출</p>
</body>
</html>
"""
soup2 = BeautifulSoup(html_data2, "html.parser")
print("title: "+soup2.find(id="title").string)
print("my: "+soup2.find(attr="my").string)

'''
 ▷ find() 메소드 사용하기 -----
title: 제목 태그
my: 원하는 자료 추출
'''

#
print("\n ▷ find_all() 메소드 사용하기 -----")
html_data3 = """
<html>
<body>
<h1 id="title">제목 태그</h1>
<p>뷰티플숲으로 읽기</p>
<p attr="my">원하는 자료 추출</p>
<div>
    <a href="http://www.naver.com">naver</a><br>
    <a href="http://www.daum.net">daum</a>
</div>
</body>
</html>
"""
soup3 = BeautifulSoup(html_data3, "html.parser")
#print(soup3.prettify()) #html 모양처럼 보기에 편하게 만들어주는 함수
links = soup3.find_all("a") #"a"태그를 전부 잡아온다.
print(links)
for i in links:
    href = i.attrs["href"]
    text = i.string
    print(href, ", ", text)

'''
 ▷ find_all() 메소드 사용하기 -----
[<a href="http://www.naver.com">naver</a>, <a href="http://www.daum.net">daum</a>]
http://www.naver.com ,  naver
http://www.daum.net ,  daum
'''

print("\n ▷ 정규표현식 사용하기 -----")
import re
links2 = soup3.find_all(href=re.compile(r"^http://"))
print(links2)
for i in links2:
    print(i.attrs["href"])
    
print()
print(soup3.find_all("p"))    
print(soup3.find_all(["p", "h1"])) # 말 그대로 다 가져온다.
aa = soup3.find_all(string=["제목 태그", "원하는 자료 추출"])
print(aa[0])
print(aa[1])

'''
 ▷ 정규표현식 사용하기 -----
[<a href="http://www.naver.com">naver</a>, <a href="http://www.daum.net">daum</a>]
http://www.naver.com
http://www.daum.net

[<p>뷰티플숲으로 읽기</p>, <p attr="my">원하는 자료 추출</p>]
[<h1 id="title">제목 태그</h1>, <p>뷰티플숲으로 읽기</p>, <p attr="my">원하는 자료 추출</p>]
제목 태그
원하는 자료 추출
'''

print("\n ▷ CSS selector 사용하기 -----")
html_data4 = """
<html>
<body>
    <div id="hello">
        <a href="http://www.naver.com">naver</a><br>
        <ul class="world">
            <li>안녕</li>
            <li>반가워</li>
        </ul>
    </div>
    <div>
        good
    </div>
</body>
</html>
"""
soup4 = BeautifulSoup(html_data4, "lxml")
a = soup4.select_one("div#hello > a").string
#div 태그 중 id=hello인 요소의 직계자손 중 a 태그를 가진 요소를 추출한다.
print("a: ", a)
uls = soup4.select("div#hello > ul.world > li")
for i in uls:
    print("li: ", i.string)

'''
 ▷ CSS selector 사용하기 -----
a:  naver
li:  안녕
li:  반가워
'''

```
<br>
###  자식, 자손, 형제
1. Children(자식): 바로 아래 태그
2. Descendants(자손): 아래에 모든 태그
3. 형제: 같은 위치에 있는 태그

```python
#scrap6.py
from urllib.request import urlopen
from bs4 import BeautifulSoup
import re

def processFunc(url):
    try:
        html = urlopen(url)
        print(html)
    except Exception as e:
        return None    
    
    try: 
        bsObj = BeautifulSoup(html, "html.parser")
        title = bsObj.body.h1
        print("연습1 : 자식과 자손태그의 차이 -----")
        #for child in bsObj.find("table", {"id":"giftList"}).children:
        for child in bsObj.find("table", {"id":"giftList"}).descendants:
            # children과 descendants는 차이가 있다.
            print("child: ", child)
        
               
        print("\n연습2 : 형제태그 -----")    
        for sibling in bsObj.find("table", {"id":"giftList"}).tr.next_siblings:
            print(sibling)
            
        print("\n연습3 : 부모(이전) 태그 -----")    
        print(bsObj.find("img", {"src":"../img/gifts/img1.jpg"}).parent.previous_sibling.get_text())
            
    except Exception as err:
        return None    
    
    return title

title = processFunc("https://www.pythonscraping.com/pages/page3.html")
if title == None:
    print("처리 실패")
    
else:
    print(title)
```
<br>
###  Scrap File저장하기
스크래핑 데이터를 파일로 저장하는 예제이다.  
```python
#scrap7.py
# 스크래핑 자료 파일로 저장
from bs4 import BeautifulSoup
import urllib.request as req
import datetime

url ="https://finance.naver.com/marketindex/"

res = req.urlopen(url)
soup = BeautifulSoup(res, "html.parser")

price = soup.select_one("span.value").string
print("usd: ", price)

t = datetime.date.today()
print(t)

fname = t.strftime("%Y-%m-%d") + ".txt"
# txt 파일로 저장가능
print(fname)
with open(fname, "w", encoding="utf-8") as f:
    f.write(price)
```
<br>
###  Scrap Image
아래 예제는 Web상의 Image를 가져오고 저장하는 예제이다.  
```python
#scrap3.py
import urllib.request
import xml.etree.ElementTree as et
# 웹 이미지 읽기
url = "https://github.com/wjddyd66/wjddyd66.github.io/blob/master/static/img/programmer.png"
save_name = "test1.png"

#다운로드
urllib.request.urlretrieve(url, save_name)
print("다운로드 후 저장 성공")

#다운로드2
save_name = "test2.png"
imsi = urllib.request.urlopen(url).read()

with open(save_name, mode="wb") as f:
    f.write(imsi)
    print("저장완료")
```
<hr>

<br>
###  Json File
<span style ="color: red">**BeautifulSoup을 통해서는 Json File을 읽어올 수 없으므로 import josn으로 module을가져와서 처리하여야 한다.**</span><br>
```python
#scrap8.py
import json
json.dumps(['foo', {'bar': ('baz', None, 1.0, 2)}])
print(json.dumps("\"foo\bar"))
print(json.dumps('\u1234'))
print(json.dumps('\\'))
print(json.dumps({"c": 0, "b": 0, "a": 0}, sort_keys=True))

print()
data = {"b":3.4, "a":0, "c":"hello world", "d":{"sbs":5}}
print(type(data))

json_data = json.dumps(data)
print(type(json_data))

print()
json_data2 = json.loads(json_data)
print(type(json_data2))

json_data = json.dumps(data, sort_keys=True)
print(json_data)

print("*"*50)
import json
json.dumps(['foo', {'bar': ('baz', None, 1.0, 2)}])
print(json.dumps("\"foo\bar"))
print(json.dumps('\u1234'))
print(json.dumps('\\'))
print(json.dumps({"c": 0, "b": 0, "a": 0}, sort_keys=True))

print()
data = {"b":3.4, "a":0, "c":"hello world", "d":{"sbs":5}}
print(type(data))

json_data = json.dumps(data)
print(type(json_data))

print()
json_data2 = json.loads(json_data)
print(type(json_data2))

json_data = json.dumps(data, sort_keys=True)
print(json_data)

print("*"*50)
json_data = {}

def readData(fileName):
    f = open(fileName, "r", encoding="utf-8")
    lines = f.read()
    f.close()
    #print(lines)
    jdata = json.loads(lines)
    return jdata
    
def main():
    json_data = readData("ftest3.json")
    #print(json_data)
    #print(type(json_data))
    d1 = json_data["직원"]["이름"]
    d2 = json_data["직원"]["직급"]
    d3 = json_data["직원"]["전화"]
    print("이름: "+d1+", 직급: "+d2+", 전화: "+d3)
    d4 = json_data["웹사이트"]["카페명"]
    d5 = json_data["웹사이트"]["userid"]
    print("카페명: "+d4+", userid: "+d5)
    

if __name__ == "__main__":
    main()

'''
"\"foo\bar"
"\u1234"
"\\"
{"a": 0, "b": 0, "c": 0}

<class 'dict'>
<class 'str'>

<class 'dict'>
{"a": 0, "b": 3.4, "c": "hello world", "d": {"sbs": 5}}
**************************************************
"\"foo\bar"
"\u1234"
"\\"
{"a": 0, "b": 0, "c": 0}

<class 'dict'>
<class 'str'>

<class 'dict'>
{"a": 0, "b": 3.4, "c": "hello world", "d": {"sbs": 5}}
**************************************************
이름: 홍길동, 직급: 과장, 전화: 111-1111
카페명: cafe.daum.net/flowlife, userid: good

'''

```

<br>
참조:<a href="https://github.com/wjddyd66/Python/tree/master/Scrapping%2CCrawling">원본코드</a><br>
코드에 문제가 있거나 궁금한 점이 있으면 wjddyd66@naver.com으로  Mail을 남겨주세요.