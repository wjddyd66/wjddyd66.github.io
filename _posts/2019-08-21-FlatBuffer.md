---
layout: post
title:  "FlatBuffer"
date:   2019-08-21 10:00:00 +0700
categories: [others]
---

### FlatBuffer
<script type="text/javascript" src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS_HTML"></script>
**FlatBuffer**란 메세지 송/수신에 사용되는 플랫폼 종속성 없이 사용가능한 "직/역직렬화 라이브러리"이다.  
**FlatBuffer**사용 이유  
1. 데이터 송/수신 시 Parsing/Unpacking이 필요 없다.
2. 메모리 효율성이 높고, 빠른 속도를 보장한다.
3. 사용하는 데이터타입에 대한 유연성이 존재한다.
4. 적은 양의 코드로 작성 가능하다.
5. 사용하기 편리하다.
6. 여러 플랫폼에서 사용가능하다.

위와 같은 장점이 존재하는 **FlatBuffer**는 직/역직렬화 라이브러리라는 점에서 **Google Protocol Buffer** 와 같은 역활을 한다.  

아래 사진을 참고하면 **FlatBuffer**를 사용하는 이유를 알 수 있다.  
<div><img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=http%3A%2F%2Fcfile3.uf.tistory.com%2Fimage%2F2243D54F582EA75B1FCD94" height="100%" width="100%" /></div><br>

**Google Protocol Buffer** 와 **FlatBuffer**의 가장 큰 차이점은 FlatBuffer의 사용 이유의 1번의 내용처럼 데이터 송/수신 시 Parsing/Unpacking이 필요 없다는 점 이다.(Zero-copy)  
이로 인하여 **시간**의 측면에서 매우 큰 장점을 나타내는 것을 보인다.  

아래 사진또한 **FlatBuffer**를 사용하는 이유를 알 수 있다.  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/AI/flatbuffer.png" height="400" width="600" /></div>
<br>

**Google Protocol Buffer**에 비해 **FlatBuffer**의 가장 뚜렷한 장점 2가지는 zero-copy와 Random-access-reads이다.  
**1) Zero Copy**  
Zero Copy는 Network에서 Read/Write할때 걸리는 불필요한 Copy과정을 최소화 하자는 것이다.  
아래 그림을 살펴보게 되면 기존의 데이터 복사 과정이다.  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/AI/flatbuffer5.png" height="400" width="600" /></div><br>
**Application buffer**를 거쳐가므로 4번의 과정이 필요하다는 것을 알 수 있다.  

아래 그림은 **Zero Copy**의 과정이다.  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/AI/flatbuffer6.png" height="400" width="600" /></div><br>
**Application buffer**를 거쳐지 않으므로 2번의 과정이라는 것을 알 수 있다.  

**2) Random-access-reads**  
**FileBuffers**는 각 레코드에 모든 필드 위치에 대한 오프셋 테이블을 저장하고 개체간 포인터를 사용하여 임의 액세스를 허용  

### 구조 및 사용 방법
**Google Protocol Buffer** 와 마찬가지로 과정이 진행된다.  
IDL(Interface Description Language)작성 -> flatc로서 Compile -> 사용
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/AI/flatbuffer2.png" height="100%" width="100%" /></div><br>


### IDL(Interface Description Language) 작성
아래와 같은 예시를 살펴보고 이것에 대한 내용을 정리한다.  
```code
// Example IDL file for our monster's schema.

namespace MyGame.Sample;

enum Color:byte { Red = 0, Green, Blue = 2 }

union Equipment { Weapon } // Optionally add more tables.

struct Vec3 {
  x:float;
  y:float;
  z:float;
}

table Monster {
  pos:Vec3;
  mana:short = 150;
  hp:short = 100;
  name:string;
  friendly:bool = false (deprecated);
  inventory:[ubyte];
  color:Color = Blue;
  weapons:[Weapon];
  equipped:Equipment;
  path:[Vec3];
}

table Weapon {
  name:string;
  damage:short;
}

root_type Monster;
```
<br>
1) Table: Field의 집합  
Field 추가 가능  
deprecated로서 사용하지 않는 Field정의 가능  
Field이름과 Table이름 바꾸기 가능  
값을 직접 대입한 뒤, Value값을 주지 않으면 Default Value로서 자동으로 들어가게 된다.  

2)  Struct: Field의 집합  
Field가 추가 되거나 deprecated할 수 없다.  
Default Value가 없기때문에 값을 반드시 대입하여야 한다.  
Table에 비해 메모리를 적게 차지하고 Access가 빠르다.  

3) Type  
**Scalar**  
- 8 bit: byte (int8), ubyte (uint8), bool
- 16 bit: short (int16), ushort (uint16)
- 32 bit: int (int32), uint (uint32), float (float32)
- 64 bit: long (int64), ulong (uint64), double (float64)

**Non Scalar**  
- Vector: [type]
- String: [byte] or [ubyte]
- tables, structs, enums, unions

**Filed 변경시 같은 크기의 Type으로만 변경 가능**  

4) (Default) Values  
숫자로 구성된 값  
Scalar만 (Default)Values를 가질 수 있고, Non Scalar는 불가능  

5) Enums  
주어진 값을 가지거나 이전 값에서 하나씩 증가하는 값을 가진 sequence  

6) Unions  
여러 Message유형을 보낼 때 사용  
Table의 이름으로서 구성  

7) Namespaces  
식별자가 고유하도록 ㄱ보장하는 코드 영역 정의  

8) Root type  
직렬화 된 데이터의 Root Table  

<br>

참조: <a href="https://google.github.io/flatbuffers/flatbuffers_guide_writing_schema.html">IDL 작성 자세한 내용</a>  


### flatc
.proto file을 **해당 언어에 맞는 클래스**로 Compile하는 과정  
1) flatbuffers 설치: <a href="https://github.com/google/flatbuffers">Download the Package</a>  
2)Building with Cmake: <a href="https://google.github.io/flatbuffers/flatbuffers_guide_building.html">Building</a>  
3) flatc 실행:  
<code>flatc [ GENERATOR OPTIONS ] [ -o PATH ] [ -I PATH ] [ -S ] FILES...
      [ -- FILES...]</code>  

GENERATOR OPTIONS을 활용하여 **해당 언어어 맞는 클래스**로서 Compile가능  
참조: <a href="https://google.github.io/flatbuffers/flatbuffers_guide_using_schema_compiler.html">Detail of Compiler Options</a>  

**실제 동작 Code**  
<code>flatc --python -o ./ ./monster.fbs</code>  
**실행 결과(monster.py)**  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/AI/flatbuffer3.png" height="100%" width="100%" /></div><br>
**Name Space**에 지정한 MyGame/Sample 폴더 안에 .py파일 생성  

### Writing a Message
1) FileBuffer생성  
2) String 존재한다면 CreateString으로 문자열 생성하고 offset을 저장  
3) Start(builder)  
4) Add(builder, value)  
5) result = End(builder)  
6) builder.Finish(result)  
7) builder.Output()  


아래 Code는 FlatBuffer Class를 사용하여서 Monster를 등록하는 과정이다.  
```python
import sys
import flatbuffers
import MyGame.Sample.Color
import MyGame.Sample.Equipment
import MyGame.Sample.Monster
import MyGame.Sample.Vec3
import MyGame.Sample.Weapon

def main():
  builder = flatbuffers.Builder(0)

#Add_weapon
  input_weapons = []
  output_weapons = []

  while True:
    weapon = raw_input("Enter a weapon name,dmage (or leave blank to finish): ")
    if weapon == "":
      break
    name, damage = weapon.split(',')
    imsi = [name, damage]
    input_weapons.append(imsi)
    
  for w in input_weapons:
    weapon_imsi = builder.CreateString(w[0])
   
    MyGame.Sample.Weapon.WeaponStart(builder)
    MyGame.Sample.Weapon.WeaponAddName(builder, weapon_imsi)
    MyGame.Sample.Weapon.WeaponAddDamage(builder, int(w[1]))
    output_weapons.append(MyGame.Sample.Weapon.WeaponEnd(builder))
    
  MyGame.Sample.Monster.MonsterStartWeaponsVector(builder, len(output_weapons))
  
  for o in output_weapons:
    builder.PrependUOffsetTRelative(o)
  # Note: Since we prepend the data, prepend the weapons in reverse order.
  weapons = builder.EndVector(len(input_weapons))

 # Add_name
  input_name = raw_input("Enter monster name: ")
  name = builder.CreateString(name)


 # Add-Inventory
  input_inventory = int(raw_input("Enter a inventory: "))
  MyGame.Sample.Monster.MonsterStartInventoryVector(builder, input_inventory)
  # Note: Since we prepend the bytes, this loop iterates in reverse order.
  for i in reversed(range(0, 10)):
    builder.PrependByte(i)
  inv = builder.EndVector(10)

 # Add-pos
  input_pos = raw_input("Enter monster pos(x,y,z): ")
  x,y,z = input_pos.split(',')
  pos = MyGame.Sample.Vec3.CreateVec3(builder, float(x), float(y), float(z))

 #Add-Hp
  input_hp = int(raw_input("Enter a hp: "))

 #Add-Color
  input_color = raw_input("Is this Monster'color Red, Green, or Blue? ")
  

 # Add-Mana
  input_mana = int(raw_input("Enter a Mana: "))


  MyGame.Sample.Monster.MonsterStart(builder)
  MyGame.Sample.Monster.MonsterAddPos(builder, pos)
  MyGame.Sample.Monster.MonsterAddMana(builder, input_mana)
  MyGame.Sample.Monster.MonsterAddHp(builder, input_hp)
  MyGame.Sample.Monster.MonsterAddName(builder, name)
  MyGame.Sample.Monster.MonsterAddWeapons(builder, weapons)
  MyGame.Sample.Monster.MonsterAddInventory(builder, inv)

  
  if input_color == "Red":
    MyGame.Sample.Monster.MonsterAddColor(builder,MyGame.Sample.Color.Color().Red)
  elif input_color == "Green":
    MyGame.Sample.Monster.MonsterAddColor(builder,MyGame.Sample.Color.Color().Green)
  elif input_color == "Blue":
    MyGame.Sample.Monster.MonsterAddColor(builder,MyGame.Sample.Color.Color().Blue)
  else:
    print("Unknown Color type; leaving as default value.(Red)")
    MyGame.Sample.Monster.MonsterAddColor(builder,MyGame.Sample.Color.Color().Red)
  
  
  MyGame.Sample.Monster.MonsterAddEquippedType(
      builder, MyGame.Sample.Equipment.Equipment().Weapon)
  print("Which weapon",input_name," equipped?")
  count = 0
  for w in input_weapons:
    print("Number: ",count,'Weapon_name: ',w[0])
    count = count+1

  select_weapon = int(raw_input("Choose Select: "))
  
  MyGame.Sample.Monster.MonsterAddEquipped(builder, output_weapons[select_weapon])
  

  orc = MyGame.Sample.Monster.MonsterEnd(builder)

  builder.Finish(orc)

  buf = builder.Output()

  with open(sys.argv[1], "wb") as f:
    f.write(buf)
  print 'The FlatBuffer was successfully created and verified!'

if __name__ == '__main__':
  main()

```
<br>

**실제 Input값**  
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
		<td>10</td>	
	</tr>

	<tr>	
		<td>Equipped</td><td>name: w1, damage: 10</td>
	</tr>

</table>
<br>


**실행 결과**  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/AI/flatbuffer4.png" height="400" width="600" /></div>
<br>
**실행 결과 Binary File로서 저장되는 것을 알 수 있다.**  

### Reading a Message
Non-deprecated Scalar 변수에 대해서는 바로 접근 가능하다.  
```python
hp = monster.Hp()
mana = monster.Mana()
name = monster.Name()
```
<br>
Sub-Object의 경우 Sub-Object에 접근 한 뒤 Sub-Object 의 Field에 접근 가능  
```python
pos = monster.Pos()
x = pos.X()
y = pos.Y()
z = pos.Z()
```
<br>
Vector의 경우 Length를 통하여 길이를 알아낸 뒤 반복문으로 접근 가능  
```python
inv_len = monster.InventoryLength()
third_item = monster.Inventory(2)
```
<br>
Union의 경우 해당 위치와 크기를 통하여 접근 가능하다.  
```python
union_type = monster.EquippedType()
if union_type == MyGame.Sample.Equipment.Equipment().Weapon:
  # `monster.Equipped()` returns a `flatbuffers.Table`, which can be used to
  # initialize a `MyGame.Sample.Weapon.Weapon()`.
  union_weapon = MyGame.Sample.Weapon.Weapon()
  union_weapon.Init(monster.Equipped().Bytes, monster.Equipped().Pos)
  weapon_name = union_weapon.Name()
  weapon_damage = union_weapon.Damage()
```
<br>
위의 Union의 접근을 통해서 아래와 같은 그림을 예상할 수 있다.  
<div><img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=http%3A%2F%2Fcfile3.uf.tistory.com%2Fimage%2F2436523E582FEC12311904" height="100%" width="100%" /></div>
<br>
즉, **FileBuffers는 offset이 어디서부터 시작되고 끝나는지를 명시함으로써 Random-acess-read가 가능하다는 것을 알 수 있다.**  
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
Monster Weapon_list
Weapon Name:  b'w2'
Weapon Damage:  20
Weapon Name:  b'w1'
Weapon Damage:  10
Monster Name b'w2'
Monster Inventory
0 1 2 3 4 5 6 7 8 9 
Monster Pos
Pos.x:  10.0
Pos.y:  20.0
Pos.z:  30.0
Monster Hp 20
Monter Color: Red
Monster Mana 10
Monster Equipped List
Weapon name:  b'w1'
Weapon damage:  10

```
<br>
참조:<a href="https://google.github.io/flatbuffers/flatbuffers_guide_tutorial.html">Writing & Reading A Message 자세한 내용</a>  
<br>
<br>
<hr>
참조: <a href="https://github.com/wjddyd66/others/tree/master/Project/flatbuffer">원본코드</a><br> 
참조: <a href="https://gompangs.tistory.com/entry/Flatbuffers-%ED%94%8C%EB%9E%AB%EB%B2%84%ED%8D%BC%EB%9E%80">곰팡이 먼지연구소</a><br>
참조: <a href="https://capnproto.org/news/2014-06-17-capnproto-flatbuffers-sbe.html">capnproto</a><br>
참조: <a href="https://skydays.tistory.com/155">skydays 블로그</a><br>
문제가 있거나 궁금한 점이 있으면 wjddyd66@naver.com으로  Mail을 남겨주세요.
