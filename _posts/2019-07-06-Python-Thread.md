---
layout: post
title:  "Python-Thread"
date:   2019-07-06 09:00:00 +0700
categories: [Python]
---

###  Python Thread
Thread란 Main Process와 병렬로 수행되는 미니 Process이다.  
Thread를 사용함으로 인하여 멀티태스킹이 가능해진다.  
<a href="https://wjddyd66.github.io/java/2019/06/15/Thread.html">Thread 자세한 내용</a>
<br><br>

Thread 기본 예제
```python
#thread1.py
#스레드 (Thread)

import threading, time

def run(id):
    for i in range(1, 5):
        print("id={}-->{}".format(id, i))
        time.sleep(0.5)
        
# 스레드를 사용하지 않은 경우: 순차적으로 수행
#run(1)
#run(2)

# 스레드를 사용한 경우: 병렬처리
th1 = threading.Thread(target=run, args=("1", ))
th2 = threading.Thread(target=run, args=("2", ))
th1.start()
th2.start()
th1.join()
th2.join()
print("프로그램 종료")
'''
id=1-->1
id=2-->1
id=1-->2
id=2-->2
id=1-->3
id=2-->3
id=2-->4
id=1-->4
프로그램 종료
'''
```
<br>
Thread를 이용한 날짜, 시간 출력하기 연습
```python
#thread2.py
# 스레드를 이용한 날짜, 시간 출력
import time
from _ast import Or
now = time.localtime()
print(now)

print("현재는 {}년 {}월 {}일 {}시 {}분 {}초".format(now.tm_year, \
        now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, \
        now.tm_sec))

import threading

def showClock():
    now = time.localtime()
    print("현재는 {}년 {}월 {}일 {}시 {}분 {}초".format(now.tm_year, \
        now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, \
        now.tm_sec))
    
def my_run():
    while True: 
        showClock() 
        time.sleep(1)
    
th = threading.Thread(target=my_run)    
th.start()
th.join()

print("Bye^0*")
'''
time.struct_time(tm_year=2019, tm_mon=7, tm_mday=6, tm_hour=17, tm_min=29, tm_sec=7, tm_wday=5, tm_yday=187, tm_isdst=0)
현재는 2019년 7월 6일 17시 29분 7초
현재는 2019년 7월 6일 17시 29분 7초
현재는 2019년 7월 6일 17시 29분 8초
현재는 2019년 7월 6일 17시 29분 9초
현재는 2019년 7월 6일 17시 29분 10초
현재는 2019년 7월 6일 17시 29분 11초
현재는 2019년 7월 6일 17시 29분 12초
현재는 2019년 7월 6일 17시 29분 13초
...
'''

```
<br>
<span style ="color: red">**예전에는 stop()이 있었으나 지금은 사라졌다.**</span>
<br>
###  Python Thread 상속 받은 Class
Class 에서 Thread를 상속받아 사용할 수 있다.  

Thread 상속 받은 Class
```python
#thread4.py
#스레드를 상속받은 클래스
import threading, time, sys

class MyThread(threading.Thread):
    def run(self):
        for i in range(1, 5):
            print("id:{}==>{}".format(self.getName(), i))
            time.sleep(0.1)
            
ths = []
for i in range(2):
    th = MyThread()
    th.start()            
    ths.append(th)
    
for th in ths:
    th.join()
    
print("Bye")

'''
id:Thread-1==>1
id:Thread-2==>1
id:Thread-1==>2
id:Thread-2==>2
id:Thread-1==>3
id:Thread-2==>3
id:Thread-1==>4
id:Thread-2==>4
Bye
'''
```
<br>
###  Python Thread 공유자원과 동기화
두개 이상의 Thread가 공유자원을 사용할 때 경쟁을 하며 충돌하는 현상이 발생할 수 있다.  
이를 방지하기 위하여 <span style ="color: red">**lock.acquire()**</span>함수를 사용한다.  

```python
#thread4.py
import threading, time

g_count = 0 #복수의 스레드에서 공유 될 변수
lock = threading.Lock()

def threadCount(id, count):
    global g_count
    for i in range(count):
        lock.acquire()
        print("id %s ==> count: %s g_count:%s" %(id, i, g_count))
        g_count = g_count + 1
        lock.release()
        
for i in range(1, 6):
    threading.Thread(target=threadCount, args=(i, 5)).start()
    
time.sleep(1)

print("final g_count: ", g_count)    
print("Bye")
'''
id 1 ==> count: 0 g_count:0
id 1 ==> count: 1 g_count:1
id 1 ==> count: 2 g_count:2
id 1 ==> count: 3 g_count:3
id 1 ==> count: 4 g_count:4
id 2 ==> count: 0 g_count:5
id 2 ==> count: 1 g_count:6
id 2 ==> count: 2 g_count:7

...

final g_count:  25
Bye
'''
```
<br>
###  Python Socket,Thread
아래 예제는 Socket, Thread를 활용하여 멀티 채팅 프로그램이다.  
Server  
```python
#char_server.py
# 멀티 채팅 프로그램용 서버
import socket
import threading

ss = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
ss.bind(("127.0.0.1", 7777))
ss.listen(5) 
print("채팅 서비스를 시작합니다.")

users=[] #유저를 담는 리스트

def ChatUser(conn):
    name = conn.recv(1024)
    data = "^_^ "+name.decode("utf-8")+"님 입장^^"
    print(data)
    
    try:
        for p in users:
            p.send(data.encode("utf-8"))
            
        while True:
            msg = conn.recv(1024)    
            data = name.decode("utf-8") + "님 메시지: " + msg.decode("utf-8")
            print("수신 결과: ", data)
            for p in users:
                p.send(data.encode("utf-8"))
    
    except:
        users.remove(conn)
        data = "ㅠ_ㅠ"+name.decode("utf-8")+"님 퇴장"
        print(data)
        if users:
            for p in users:
                p.send(data.encode("utf-8"))
        else: 
            print("EXIT")
            
while True:
    conn, addr = ss.accept()
    users.append(conn)
    th = threading.Thread(target=ChatUser, args=(conn,))
    th.start()
```
<br>
Client
```python
#char_client.py
# 멀티 채팅 프로그램용 서버
import socket
import threading
import sys

def handle(socket):
    while True:
        data = socket.recv(1024)
        if not data:continue
        print(data)

sys.stdout.flush() #flush(): 깨끗하게 비운다.

name = input("채팅에 사용 할 아이디를 입력하세요: ")
cs = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
cs.connect(("127.0.0.1", 7777))
cs.send(name.encode("utf-8"))

th = threading.Thread(target=handle, args=(cs,))
th.start()

while True:
    msg = input()
    sys.stdout.flush()
    if not msg:continue
    cs.send(msg.encode("utf-8"))
    
cs.close()    
```
결과화면
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Python/Thread1.PNG" height="200" width="600" /></div>
<br>
<span style ="color: red">**한글을 사용하기 위해서는 utf-8로서 Encoding하는 작업이 필요하다.**</span><br>
<hr>

참조:<a href="https://github.com/wjddyd66/Python/tree/master/Thread">원본코드</a><br>
코드에 문제가 있거나 궁금한 점이 있으면 wjddyd66@naver.com으로  Mail을 남겨주세요.