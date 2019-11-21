---
layout: post
title:  "Google Protocol Buffer3"
date:   2019-08-21 09:30:00 +0700
categories: [others]
---

### Google Protocol Buffer3
<script type="text/javascript" src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS_HTML"></script>
Google Protocol Buffer3에 대해 자세히 알아보기 전에 한가지 문제에 대해 알아보자.  
**Structured Data**를 어떻게 Serilalize 하고 Retrieve할수 있을까?  
Google Protocol Buffer3를 사용하기 이전에는 다음과 같은 방법과 그러한 방법에 대한 단점이 있다.  
**기존 방식 3가지**  
<table class="table">

	<tr>	
		<td>방법</td><td>특징</td>
	</tr>
	
	<tr>	
		<td>Python pickling</td>
		<td>
		<ul>
		<li>다른 언어와의 호환이 잘 안됨</li>
		</ul>		
		</td>
	</tr>

	<tr>	
		<td>Encode the data items into a Single String</td>
		<td>
		<ul>
		<li>small run-time cost필요</li>
		<li>Simple Data Encoding에서 활용</li>
		</ul>		
		</td>
	</tr>

	<tr>	
		<td>XML</td>
		<td>
		<ul>
		<li>Dom형식이므로 접근 가능</li>
		<li>Encoding, Decoding에 많은 시간 필요</li>
		</ul>		
		</td>	
	</tr>
</table>
<br>

**Google Protocol Buffer3**는 프로토콜 버퍼 데이터의 자동 인코딩 및 구문 분석을 구현하는 클래스를 만들어서 접근 가능  
Getter 와 Setter를 제공하고 프로토콜 버퍼를 단위로 읽고 쓰는 세부사항 처리 가능  
직렬화 데이터 구조(Serialized Data Structure)를 C++, C#, Java, Python등 다양한 언어를 지원하며 특히 직렬화 속도가 빠르고 파일의 크기가 작다.  

### 구조 및 사용 방법
프로토콜 버퍼를 사용하기 위해서는 저장하기 위한 **데이터형**을 proto file이라는 형태로 정의한다.  
**proto file**의 특징은 **특정 언어에 종속성이 없는 형태**로 데이터 타입을 정의한다는 것이다.  
이렇게 정의된 데이터 타입을 프로그래밍 언어에 사용하려면, **해당 언어에 맞는 클래스**로 생성해야 한다.  
**protoc 컴파일러로 proto file을 컴파일 하면, 각 언어에 맞는 형태의 데이터 클래스 파일을 생성**해 준다.  
<div><img src="https://t1.daumcdn.net/cfile/tistory/233E2635594F907222" height="100%" width="100%" /></div><br>


### Proto file
.proto file은 각 **Field**에 대한 **Name과 Type**으로 지정된다.  
아래와 같은 예시를 살펴보고 이것에 대한 내용을 정리한다.  
```code
syntax = "proto3";
package MyGame.Sample;

message Monster {
  Vec3 pos = 1;
  int32 mana = 2;
  int32 hp = 3;
  string name = 4;
  bool friendly = 5;
  repeated int32 inventory = 6;
  Color color= 7;
  repeated Weapon weapons =8;
  Equipment equipped =9;
  repeated Vec3 path =10;



 enum Color{
 Red = 0;
 Green = 1;
 Blue = 2;
}
}

message Vec3{
 float x = 1;
 float y = 2;
 float z = 3;
}

 message Equipment{
 repeated Weapon weapon = 1;
}

message Weapon{
 string name = 1;
 int32 damage = 2;
}

message AddressMonster{
 repeated Monster monster = 1;
}
```
<br>
**.proto file 정의 방법**  
1) Package: 다른 프로젝트 간의 이름 충돌 방지  
**Python 에서 패키지는 일반적으로 디렉토리 구조에 영향을 받으니 파이썬이 아닌 언어에서 Name Space충돌을 피하기 위하여 선언**  
2) Message: Field의 집합  
3) Type  
<table class="table">

	<tr>	
		<td>.proto Type</td><td>Python Type</td>
	</tr>

	<tr>	
		<td>double</td><td>float</td>
	</tr>
	
	<tr>	
		<td>float</td><td>float</td>
	</tr>

	<tr>	
		<td>int32</td><td>int</td>
	</tr>

	<tr>	
		<td>int64</td><td>int/long</td>
	</tr>

	<tr>	
		<td>unit32</td><td>int/long</td>
	</tr>

	<tr>	
		<td>unit64</td><td>int/long</td>
	</tr>

	<tr>	
		<td>sint32</td><td>int</td>
	</tr>

	<tr>	
		<td>sint64</td><td>int/long</td>
	</tr>

	<tr>	
		<td>fixed32</td><td>int/long</td>
	</tr>

	<tr>	
		<td>fixed64</td><td>int/long</td>
	</tr>

	<tr>	
		<td>sfixed32</td><td>int</td>
	</tr>

	<tr>	
		<td>sfixed64</td><td>int/long</td>
	</tr>

	<tr>	
		<td>bool</td><td>bool</td>
	</tr>

	<tr>	
		<td>string</td><td>str</td>
	</tr>

	<tr>	
		<td>bytes</td><td>bytes</td>
	</tr>

	<tr>	
		<td>다른 message</td><td></td>
	</tr>
</table>
<br>
4) "=1", "=2", ...  
Unique한 Tag를 Field에 대입하는 과정이다.  
- 1~15: Commonly used or repeate elements
- 16이상: loss-commonly used

<br>
5) Annotation of filed  
- required: 반드시 Value필요
- optional: Value는 필요 없으면, Value를 정의안하면 Default value사용
- repeated: Scalar가아닌 Vector로서 표현

<br>

참조: <a href="https://developers.google.com/protocol-buffers/docs/proto">.proto Type자세한 내용</a>  


### protoc(Compling Protocol Buffers)
.proto file을 **해당 언어에 맞는 클래스**로 Compile하는 과정  
1) Compiler 설치: <a href="https://developers.google.com/protocol-buffers/docs/downloads">Download the Package</a>  
2) Compiler 실행:  
<code>protoc -I=$SRC_DIR --python_out=$DST_DIR $SRC_DIR/addressbook.proto</code>  
option을 활용하여 **해당 언어어 맞는 클래스**로서 Compile가능

**실제 동작 Code**  
<code>protoc -I=. --python_out=. monster.proto</code>  
**실행 결과(monster.py)**  

```python
import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='monster.proto',
  package='MyGame.Sample',
  syntax='proto3',
  serialized_pb=_b('\n\rmonster.proto\x12\rMyGame.Sample\"\xc3\x02\n\x07Monster\x12 \n\x03pos\x18\x01 \x01(\x0b\x32\x13.MyGame.Sample.Vec3\x12\x0c\n\x04mana\x18\x02 \x01(\x05\x12\n\n\x02hp\x18\x03 \x01(\x05\x12\x0c\n\x04name\x18\x04

...

AddressMonster = _reflection.GeneratedProtocolMessageType('AddressMonster', (_message.Message,), dict(
  DESCRIPTOR = _ADDRESSMONSTER,
  __module__ = 'monster_pb2'
  # @@protoc_insertion_point(class_scope:MyGame.Sample.AddressMonster)
  ))
_sym_db.RegisterMessage(AddressMonster)


# @@protoc_insertion_point(module_scope)

```
### Writing a Message
1) monster_pb2.Monster() 로서 monster 객체 생성  
2) 객체의 입력받은 값 넣기  
3) monster_pb2.AddressMonster()로서 monster추가  
4) SerializeToString()을 통하여 문자열을 직렬화  

아래 코드는 Protocol Buffer Class를 사용하여서 Monster를 등록하는 과정이다.  

```python
#!/usr/bin/env python
# coding: utf-8

import monster_pb2
import sys

try:
  raw_input          # Python 2
except NameError:
  raw_input = input  # Python 3


def PromptForMonster(monster):
    monster.mana = int(raw_input("Enter monster mana: "))
    monster.hp = int(raw_input("Enter monster hp: "))
    monster.name = raw_input("Enter monster name: ")
    monster.friendly = bool(raw_input("Enter monster friendly: "))
    weapons =[]
    
    while True:
        weapon = raw_input("Enter a weapon name,dmage (or leave blank to finish): ")
        if weapon == "":
          break
        name, damage = weapon.split(',')
        imsi = [name, damage]
        weapons.append(imsi)
        monster_weapon = monster.weapons.add()
        monster_weapon.name = name
        monster_weapon.damage = int(damage)
        
    monster_inven = []
    while True:
        inven = raw_input("Enter a inventory (or leave blank to finish): ")
        if inven == "":
          break
        
        monster_inven.append(int(inven))
        
    monster.inventory[:] = monster_inven
            
    type = raw_input("Is this a Red, Green, or Blue? ")
    if type == "Red":
      monster.color = monster_pb2.Monster.Red
    elif type == "Green":
      monster.color = monster_pb2.Monster.Green
    elif type == "Blue":
      monster.color = monster_pb2.Monster.Blue
    else:
      print "Unknown Color type; leaving as default value."
    
    
    pos = raw_input("Enter monster pos(x,y,z): ")
    x,y,z = pos.split(',')
    monster_pos = monster.pos
    monster_pos.x = int(x)
    monster_pos.y = int(y)
    monster_pos.z = int(z)
    
    while True:
        path = raw_input("Enter a path(x,y,z) (or leave blank to finish): ")
        if path == "":
          break
        
        x,y,z = path.split(',')
        monster_path = monster.path.add()
        monster_path.x = int(x)
        monster_path.y = int(y)
        monster_path.z = int(z)
    
    equip = monster.equipped
    for w in weapons:
        equip.weapon.add(name=w[0],damage=int(w[1]))
        

if len(sys.argv) != 2:
  print("Usage:", sys.argv[0], "SAVE_FILE_NAME")
  sys.exit(-1)


address_monster = monster_pb2.AddressMonster()


try:
  with open(sys.argv[1], "rb") as f:
    address_monster.ParseFromString(f.read())
except IOError:
  print(sys.argv[1] + ": File not found.  Creating a new file.")



PromptForMonster(address_monster.monster.add())


with open(sys.argv[1], "wb") as f:
  f.write(address_monster.SerializeToString())

```
<br>

**실제 Input값**  
**Monster1**
<table class="table">

	<tr>	
		<td>Variable</td><td>실제값</td>
	</tr>
	
	<tr>	
		<td>name</td><td>monster1</td>
	</tr>

	<tr>	
		<td>mana</td><td>10</td>
	</tr>

	<tr>	
		<td>hp</td><td>20</td>
	</tr>

	<tr>	
		<td>friendly</td><td>True</td>
	</tr>

	<tr>	
		<td>Color</td><td>Red</td>
	</tr>

	<tr>	
		<td>Pos</td><td>10, 20, 30</td>
	</tr>

	<tr>	
		<td>weapons</td>
		<td>
		<ul>
		<li>name: w1, damage: 10</li>
		<li>name: w2, damage: 20</li>
		</ul>		
		</td>
	</tr>

	<tr>	
		<td>inventory</td>
		<td>
		<ul>
		<li>1</li>
		<li>2</li>
		<li>3</li>
		</ul>		
		</td>	
	</tr>

	<tr>	
		<td>path</td>
		<td>
		<ul>
		<li>1,2,3</li>
		<li>3,2,1</li>
		</ul>		
		</td>
	</tr>
</table>
<br>

**Monster2**
<table class="table">

	<tr>	
		<td>Variable</td><td>실제값</td>
	</tr>
	
	<tr>	
		<td>name</td><td>monster2</td>
	</tr>

	<tr>	
		<td>mana</td><td>100</td>
	</tr>

	<tr>	
		<td>hp</td><td>200</td>
	</tr>

	<tr>	
		<td>friendly</td><td>False</td>
	</tr>

	<tr>	
		<td>Color</td><td>Green</td>
	</tr>

	<tr>	
		<td>Pos</td><td>100, 200, 300</td>
	</tr>

	<tr>	
		<td>weapons</td>
		<td>
		<ul>
		<li>name: w3, damage: 10</li>
		<li>name: w4, damage: 20</li>
		</ul>		
		</td>
	</tr>

	<tr>	
		<td>inventory</td>
		<td>
		<ul>
		<li>3</li>
		<li>2</li>
		<li>1</li>
		</ul>		
		</td>	
	</tr>

	<tr>	
		<td>path</td>
		<td>
		<ul>
		<li>10,20,30</li>
		<li>30,20,10</li>
		</ul>		
		</td>
	</tr>
</table>
<br>

**실행 결과**  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/AI/googleprotocol.png" height="400" width="600" /></div>
<br>
**실행 결과 Binary File로서 저장되는 것을 알 수 있다.**  

### Reading a Message
ParseFromString()을 통하여 문자열을 객체로서 접근 가능하게 한다.  
아래 코드는 만들어진 Binary File을 읽는 과정이다.  
```python
#!/usr/bin/env python
# coding: utf-8


import monster_pb2
import sys


def ListMonster(address_monster):
  for monster in address_monster.monster:
    print("Monster pos: ",monster.pos.x, monster.pos.y, monster.pos.z)
    print("Monster mana:", monster.mana)
    print("Monster hp:", monster.hp)
    print("Monster name:", monster.name)
    print("Monster friendly:", monster.friendly)
    
    print("Monster inventory: ")
    for monster_inventory in monster.inventory:
        print(monster_inventory)
    
    print("Monster color: ",monster.color)
    
    print("Monster weapon: ")
    for w in monster.weapons:
        print("Name: ",w.name)
        print("Damage: ",w.damage)
    
    print("Monster equipped")
    for e in monster.equipped.weapon:
        print(e)
    
    print("Monster path:")
    for p in monster.path:
        print(p)


if len(sys.argv) != 2:
  print("Usage:", sys.argv[0], "Save_FILE")
  sys.exit(-1)

address_monster = monster_pb2.AddressMonster()


with open(sys.argv[1], "rb") as f:
  address_monster.ParseFromString(f.read())

ListMonster(address_monster)

```
<br>
**실행 결과**  
```code
Monster pos:  10.0 20.0 30.0
Monster mana: 10
Monster hp: 20
Monster name: monster1
Monster friendly: True
Monster inventory: 
1
2
3
Monster color:  0
Monster weapon: 
Name:  w1
Damage:  10
Name:  w2
Damage:  20
Monster equipped
name: "w1"
damage: 10

name: "w2"
damage: 20

Monster path:
x: 1.0
y: 2.0
z: 3.0

x: 3.0
y: 2.0
z: 1.0

Monster pos:  100.0 200.0 300.0
Monster mana: 100
Monster hp: 200
Monster name: monster2
Monster friendly: True
Monster inventory: 
3
2
1
Monster color:  1
Monster weapon: 
Name:  w3
Damage:  10
Name:  w4
Damage:  20
Monster equipped
name: "w3"
damage: 10

name: "w4"
damage: 20

Monster path:
x: 10.0
y: 20.0
z: 30.0

x: 30.0
y: 20.0
z: 10.0


```
<br>
<br>
<hr>
참조: <a href="https://github.com/wjddyd66/others/tree/master/Project/google_protocol_buffer_3">원본코드</a><br> 
참조: <a href="https://bcho.tistory.com/1182">조대협의 블로그</a><br>
문제가 있거나 궁금한 점이 있으면 wjddyd66@naver.com으로  Mail을 남겨주세요.
