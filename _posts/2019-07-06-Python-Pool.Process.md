---
layout: post
title:  "Python-Pool,Process"
date:   2019-07-06 09:30:00 +0700
categories: [Python]
---

###  Python Thread의 한계
Python은 Interpreter 언어이기 때문에 Compile언어보다 느리고 따라서 실시간 거래 시스템처럼 매우 짧은 응답시간을 필요로 하는데 사용할 수 없다.   
다시 말해서 파이썬은 동시다발적인 멀티스레드를 처리하거나 CPU에 집중된 많은 스레드를 처리하는 데 적합하지 않다.  
이는 바로 GIL(Global Interpreter Lock) 때문이다. 이 메커니즘은 인터프리터가 한 번에 하나의 바이크 코드명령만 실행하도록 하는 것을 말하는데  프로그래머는 만들고자 하는 프로그램의 대부분을 파이썬으로 만들 수 있지만 시스템 프로그래밍이나 하드웨어 제어와 같은 매우 복잡하고 반복 연산이 많은 프로그램은 만들 수 없는 것이 그 이유이다.  
출처: <a href="https://wangin9.tistory.com/entry/pythonthreadGIL">잉구블로그</a>  



<span style ="color: red">**Python에서 Thread는 병렬 식이 아닌 일정 시간 동안 작동할 때는 다른 Thread의 작동이 멈춘다.**</span><br>
<span style ="color: red">**Python에서 병렬처리를 원하면 Pool과 Process를 이용하여 병렬구조로 처리해야 한다.**</span><br>

###  Python Pool, Process
<span style ="color: red">**Pool: 입력 값을 process에 분배하여 함수실행의 병렬화하는 편리한 수단을 제공**</span><br>
<span style ="color: red">**Process: 하나의 프로세스를 하나의 함수에 적당한 인자값을 할당 또는 할당하지 않고 실행**</span><br>

Pool
```python
#Pool.py
# <Multi Processing을 지원하는 'Pool', 'Process'로 멀티태스킹 구현>
from multiprocessing import Pool
import time
import os

def func(x):
    print("값", x, "에 대한 작업 pid=", os.getpid())
    time.sleep(1)
    return x * x

if __name__ == "__main__":
    p = Pool(3)
    startTime = int(time.time())
    """
    for i in range(0, 10):
        print(func(i))
    """    
    print(p.map(func, range(0, 10)))
    
    endTime = int(time.time())
    print("총 작업시간: ", (endTime - startTime))
    
'''
값 0 에 대한 작업 pid= 7656
값 1 에 대한 작업 pid= 13364
값 2 에 대한 작업 pid= 9436
값 3 에 대한 작업 pid= 7656
값 4 에 대한 작업 pid= 13364
값 5 에 대한 작업 pid= 9436
값 6 에 대한 작업 pid= 7656
값 7 에 대한 작업 pid= 13364
값 8 에 대한 작업 pid= 9436
값 9 에 대한 작업 pid= 7656
[0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
총 작업시간:  4 
'''
```
<br>

Process
```python
#Process.py
# Process: Pool과는 달리, 하나의 프로세스를 하나의 함수에 할당해주는 방식 (건너건너X)

import os
from multiprocessing import Process

def func():
    print("멀티 처리를 하고 싶은 내용 기술")
    
def doubler(num):
    result = num + 10
    func()
    proc = os.getpid()
    print("num:{}, result:{}, process:{}".format(num, result, proc))    
    
if __name__ == "__main__":
    nums = [1,2,3,4,5]
    procs = []    
    
    for i, number in enumerate(nums):
        proc = Process(target=doubler, args=(number,))
        procs.append(proc)
        proc.start()
        
    for proc in procs:
        proc.join()

'''
num:1, result:11, process:8160
num:5, result:15, process:12584
멀티 처리를 하고 싶은 내용 기술
num:2, result:12, process:4952
멀티 처리를 하고 싶은 내용 기술
num:3, result:13, process:13444
멀티 처리를 하고 싶은 내용 기술
num:4, result:14, process:32
'''
```
<br>
<span style ="color: red">**MultiProcessing을 통해 Process를 많이 쓸수록 속도가 빨라지나 CPU의 성능을 고려해서 사용할 Process의 개수를 조절하여 한다. (일반적으로는 3~5개)**</span><br>
<hr>

참조:<a href="https://github.com/wjddyd66/Python/tree/master/MultiProcess">원본코드</a><br>
코드에 문제가 있거나 궁금한 점이 있으면 wjddyd66@naver.com으로  Mail을 남겨주세요.