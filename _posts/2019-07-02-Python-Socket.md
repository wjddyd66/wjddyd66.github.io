---
layout: post
title:  "Python-Socket"
date:   2019-07-02 09:00:00 +0700
categories: [Python]
---

###  Python Socket
TCP 서버/클라이언트 함수 호출 관계  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Python/Soc3.PNG" height="100%" width="100%" /></div>
그림 출처:<a href="http://blog.naver.com/PostView.nhn?blogId=cnfldidhd&logNo=20171560152">젤리 블로그</a>  
<br><br>
TCP 서버/클라이언트 함수 호출 순서  

1. 서버, 클라이언트 소켓 생성
2. 서버는 bind(), listen()함수를 호출하여 대기 상태
3. 클라이언트는 connect()함수 호출을 통해 연결 요청
4. 서버는 accept()함수를 통해 연결 수락
5. 서버와 클라이언트는 서로 연결
6. close()함수를 통해 연결 종료

<br><br>
<span style ="color: red">**Socket 함수**</span><br> 
<link rel = "stylesheet" href ="/static/css/bootstrap.min.css">

<table class="table">
	<tbody>
	<tr>
		<td>명령어</td><td>설명</td>
	</tr>
	<tr>
		<td>socket()</td><td>소켓 생성</td>
	</tr>
		<tr>
		<td>bind()</td><td>생성한 socket을 server socket 으로 등록</td>
	</tr>
		<tr>
		<td>listen()</td><td>server socket을 통해 클라이언트의 접속 요청을 확인하도록 설정</td>
	</tr>
			<tr>
		<td>accept()</td><td>클라이언트 접속 요청 대기 및 허락<br>클라이언트와 통신을 위해 새 socket생성</td>
	</tr>
				<tr>
		<td>read(),wirte()</td><td>client socket으로 데이터를 송수신</td>
	</tr>
					<tr>
		<td>close()</td><td>client socket을 소멸</td>
	</tr>
	</tbody>
</table>
<br>
Server
```python
#soc1_server.py
#<컴퓨터 간 접속상태 확인을 위해 1회 접속처리>
from socket import *

# TCP/IP socket
serverSock = socket(AF_INET, SOCK_STREAM)
serverSock.bind(("127.0.0.1", 9999))
serverSock.listen(1) #리스너 설정 1~5
print("server service 중 ...")

conn, addr = serverSock.accept()
print("client address: ", addr)
print("from client message: ", conn.recv(1024).decode())

conn.close()
serverSock.close()

```
<br>
Client
```python
#soc1_client.py
from socket import *

clientSock = socket(AF_INET, SOCK_STREAM) #소켓의 종류와 유형 선언, 가장 일반적인 모습
clientSock.connect(("127.0.0.1", 9999))
clientSock.sendall("안녕!".encode(encoding="utf-8", errors="strict"))
clientSock.close()

```
<br>
실행 결과  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Python/Soc1.PNG" height="200" width="400" /></div>
<br>
###  Python Echo Server
클라이언트가 전송해주는 데이터를 그래도 되돌려 전송해 주는 기능의 서버  
Server
```python
#soc2_server.py
#echo server 사용
import socket
import sys

HOST = ""
PORT = 8888

serverSock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

try:
    serverSock.bind((HOST, PORT))
    #HOST에 IP를 별도로 주지 않는다면 동적이 된다.
    print("서버 서비스 중입니다.")
    serverSock.listen(5) # 동시 최대 접속 수=5
    
    #서버는 무한루프에 빠져있다.
    while True:
        conn, addr = serverSock.accept()
        print("client info: ", addr[0], addr[1])
        #IP주소와 포트번호를 따로 받겠다는 의미
        print("from client message: ", conn.recv(1024).decode())
        
        #송신
        conn.send(("from server: " + str(addr[1]) + \
                   "너도 잘 지내라").encode("utf-8"))
                    # \ -> 명령이 계속 이어진다는 의미
    
except Exception as e:
    print("err: ", e)
    sys.exit() #프로그램 강제종료
finally:
    serverSock.close()    
    conn.close()

```
<br>
Client
```python
#soc2_client.py
from socket import *

clientSock = socket(AF_INET, SOCK_STREAM) #소켓의 종류와 유형 선언, 가장 일반적인 모습
clientSock.connect(("127.0.0.1", 8888))
clientSock.sendall("스승의 은혜는 하늘 같아서 . . .".encode(encoding="utf-8", errors="strict"))
#errors="strict" -> 줘도, 안줘도 된다.

re_message = clientSock.recv(1024).decode()
print("수신자료: ", re_message)

clientSock.close()

```
<br>

실행 결과  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Python/Soc2.PNG" height="200" width="400" /></div>

<br>
<hr>

참조:<a href="https://github.com/wjddyd66/Python/tree/master/Socket">원본코드</a><br>
코드에 문제가 있거나 궁금한 점이 있으면 wjddyd66@naver.com으로  Mail을 남겨주세요.