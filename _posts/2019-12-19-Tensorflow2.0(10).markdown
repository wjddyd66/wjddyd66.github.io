---
layout: post
title:  "tf.function"
date:   2019-12-19 09:50:20 +0700
categories: [Tnesorflow2.0]
---
<script type="text/javascript" src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS_HTML"></script>
### tf.function
Tensorflow 2.0에 맞게 다시 Tensorflow를 살펴볼 필요가 있다고 느껴져서 <a href="https://www.tensorflow.org/?hl=ko">Tensorflow 정식 홈페이지</a>에 나와있는 예제부터 전반적인 Tensorflow 사용법을 먼저 익히는 Post가 된다.  
<br>

#### 필요한 Library Import
```python
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import traceback
import contextlib

# Gpu 사용 가능 여부 출력
print(tf.test.is_gpu_available())
```
<br>
True  
<br><br>

#### What is tf.function
Tensorflow 2.0에서는 <a href="https://wjddyd66.github.io/tnesorflow2.0/Tensorflow2.0(6)/">Eager execution</a>을 기본적으로 사용한다.  
이러한 Eager execution사용으로 인하여 사용자는 쉽게 Debug하면서 Code를 사용할 수 있으나 **성능과 배포(Deployment)가 저하될 수 있다.**  

**<code>tf.function</code>이란 위와같은 상황에서 성능을 향상시키고 Model Deployment를 할 수 있다. 또한 Python code와 함께 동작한다.**  
단 Guide에서는 다음과 같은 주의사항이 있다고 설명하고 있다.  
>- Don't rely on Python side effects like object mutation or list appends.
- tf.function works best with TensorFlow ops, rather than NumPy ops or Python primitives.
- When in doubt, use the for x in y idiom
>

위와 같은 주의사항과 왜 <code>tf.function</code>사용을 권장하는지 알아보자.  

<br><br>

### Basics
<code>tf.function</code>은 Tensorflow operation과 같이 정의할 수 있다.  
이렇게 정의된 Function은 Eager Execution과 <a href="https://wjddyd66.github.io/tnesorflow2.0/Tensorflow2.0(6)/#gradienttape2">tf.GradientTape()</a>에서 바로 사용 가능하다. 또한 Functions inside Functions또한 가능하다.

```python
# tf.function 선언
@tf.function
def add(a,b):
    return a+b

# Eager Execution
add_result = add(tf.ones([2,2]),tf.ones([2,2]))
print('Eager Execution Result')
print(add_result.numpy())
print()

# tf.GradientTape()
v = tf.Variable(1.0)
with tf.GradientTape() as tape:
    result = add(v,1.0)
gradient_result = tape.gradient(result,v)
print('Gradient Result')
print(gradient_result.numpy())
print()

# Functions inside Functions
@tf.function
def dense_layer(x,w,b):
    return add(tf.matmul(x,w),b)
function_inside_function_result = dense_layer(tf.ones([3,2]),
                                              tf.ones([2,2]),
                                              tf.ones([2]))
print('Functions inside Functions Result')
print(function_inside_function_result.numpy())
```
<br>
```code
Eager Execution Result
[[2. 2.]
 [2. 2.]]

Gradient Result
1.0

Functions inside Functions Result
[[3. 3.]
 [3. 3.]
 [3. 3.]]
```
<br>
<br><br>

### Tracing and polymorphism
#### polymorphism
Python은 dynamic typing이다. 한가지 예를 들어보면 Python에는 변수선언을 할때 int나 double, string 등으로 변수의 Type을 지정하지 않는다.  
Python 내부에서 알아서 알맞게 지정하기 때문이다.  

Tensorflow의 기초가 되는 <a href="https://wjddyd66.github.io/tnesorflow2.0/Tensorflow2.0(7)/">Tensor</a> 는 정의할때 반드시 Shape(Dimension)과 자료형(Dtype)이 필요로 하다.  
위와 같은 python의 특징을 사용하면 하나의 Function에서 서로다른 Dtype을 받아들이고 처리하는 Function을 정의할 수 있다.

```python
# Polymorphism
@tf.function
def double(a):
    print("Tracing with", a)
    return a + a

print(double(tf.constant(1)))
print()
print(double(tf.constant(1.1)))
print()
print(double(tf.constant("a")))
print()
```
<br>
```code
Tracing with Tensor("a:0", shape=(), dtype=int32)
tf.Tensor(2, shape=(), dtype=int32)

Tracing with Tensor("a:0", shape=(), dtype=float32)
tf.Tensor(2.2, shape=(), dtype=float32)

Tracing with Tensor("a:0", shape=(), dtype=string)
tf.Tensor(b'aa', shape=(), dtype=string)
```
<br>
<br><br>

#### Tracing
위에서 <code>tf.function</code>으로서 Polymorphism을 확인하였다.  
Tracing이란 <code>tf.function</code>로서 원하는 Assert 구문을 만들어서 원하는 형태로 흘러가는지 확인하는 것이다.  
Tracing의 순서는 다음과 같다.  
- <code>tf.function</code> 선언
- <code>get_concrete_function</code>으로서 Specific Trace 정의
- <code>input_signature</code>으로서 tf.function이 호출될때 Error 확인

```python
# Error 발생시 지정한 Error_class면 출력
@contextlib.contextmanager
def assert_raises(error_class):
    try:
        yield
    except error_class as e:
        print('Caught expected exception \n  {}:'.format(error_class))
        traceback.print_exc(limit=2)
    except Exception as e:
        raise e
    else:
        raise Exception('Expected {} to be raised but no error was raised!'.format(
            error_class))

# get_concrete_function으로서 Specific Trace정의
# Input Tensor의 Dtype이 String인지 확인한다.
double_strings = double.get_concrete_function(tf.TensorSpec(shape=None, dtype=tf.string))

print('Executing traced function')
print(double_strings(tf.constant('a')))
print(double_strings(tf.constant('b')))
print()
# InvalidArgumentError 발생시 Error Message 확인
print('Check InvalidArgumentError')
with assert_raises(tf.errors.InvalidArgumentError):
    double_strings(tf.constant(1))
print()

# input_signature로서 Tensor가 1Dimension인지 확인
@tf.function(input_signature=(tf.TensorSpec(shape=[None], dtype=tf.int32),))
def next_collatz(x):
    print("Tracing with", x)
    return tf.where(x % 2 == 0, x // 2, 3 * x + 1)

# 1 Dimension이므로 Error 발생 X
print(next_collatz(tf.constant([1, 2])))
# 2 Dimension이므로 Error 발생 O
print('Check ValueError')
with assert_raises(ValueError):
    next_collatz(tf.constant([[1, 2], [3, 4]]))
```
<br>
```code
Tracing with Tensor("a:0", dtype=string)
Executing traced function
tf.Tensor(b'aa', shape=(), dtype=string)
tf.Tensor(b'bb', shape=(), dtype=string)

Check InvalidArgumentError
Caught expected exception 
  <class 'tensorflow.python.framework.errors_impl.InvalidArgumentError'>:

Tracing with Tensor("x:0", shape=(None,), dtype=int32)
tf.Tensor([4 1], shape=(2,), dtype=int32)
Check ValueError
Caught expected exception 
  <class 'ValueError'>:
  
...

ValueError: Python inputs incompatible with input_signature:
  inputs: (
    tf.Tensor(
[[1 2]
 [3 4]], shape=(2, 2), dtype=int32))
  input_signature: (
    TensorSpec(shape=(None,), dtype=tf.int32, name=None))
```
<br>
<br><br>

### Python or Tensor args?
기본적으로 Hyperparameter를 Python의 Argument를 사용하여 지정하였다. 예를들어, dropout_ratio=0.1, learning_rate=0.2처럼 선언하였다.  
이러한 Argument가 바뀌게 되면 Static한 Tensorflow Graph를 재정의 하고 Retrace해야 하므로 비효율적이다.(단, Argument가 같으면 새로운 Graph생성 X)  
**Tensor args를 활용하여 Graph의 Hyperparameter를 정의하고 사용하게 되면 Tensorflow의 Graph인 AutoGraph는 Dynamical unroll이 될 것이고 다양한 traces에도 불고하고 Graph는 즉시 정의될 것이다.**  
아래 예시를 살펴보면 매우 잘 와닿을 수 있다.  
Python Argument를 사용하여 Graph를 2번 정의했을 경우 2개의 Graph가 생성되는 것을 볼 수 있다.  
Tensor Argument를 사용하면 Graph를 2번 정의하는 것이 아닌 마지막의 TensorArgument를 통하여 정의되게 된다.  
Tensor의 특징이다. Tensor Argument를 통하여 Tensorflow의 Graph를 생성하게 되면 하나의 Graph에서 Tensor를 올려두기 때문에 올려둔 Tensor의 값만 바꾸는 거지 Graph를 새롭게 그리는 것이 아니다.

```python
def train_one_step():
    pass

@tf.function
def train(num_steps):
    print(type(num_steps))
    tf.print(num_steps)
    print('Tracing with num_steps={}'.format(num_steps))
    for _ in tf.range(num_steps):
        train_one_step()

# Python Argument 사용
print('Python Argument')
train(num_steps=10)
# 값이 같은 Argument사용시 Graph생성 X
train(num_steps=10)
train(num_steps=20)
print()

# Tensor Argument 사용
# Python Argument를 Tensor로 Casting하여 사용
print('Tensor Argument')
# Graph는 한번만 생성되고 Graph안에서의 Tensor값만 바뀌게 된다.
train(num_steps=tf.constant(10))
train(num_steps=tf.constant(10))
train(num_steps=tf.constant(20))
```
<br>
```code
Python Argument
<class 'int'>
Tracing with num_steps=10
10
10
<class 'int'>
Tracing with num_steps=20
20

Tensor Argument
<class 'tensorflow.python.framework.ops.Tensor'>
Tracing with num_steps=Tensor("num_steps:0", shape=(), dtype=int32)
10
10
20
```
<br>
<br><br>

### Beware of Python state
Generator나 iterator아 같은 연산자들은 Python runtime에만 의존하게 된다. 따라서 **Eager Execution으로 살펴보는 경우 문제가 발생하지 않지만 실제 <code>tf.function</code>안에서 사용하는 예상치못한 문제가 발생할 수 있다.**  

```python
external_var = tf.Variable(0)

@tf.function
def buggy_consume_next(iterator):
    external_var.assign_add(next(iterator))
    tf.print("Value of external_var:", external_var)

iterator = iter([0, 1, 2, 3])

print('tf.function iterator')
buggy_consume_next(iterator)
# This reuses the first value from the iterator, rather than consuming the next value.
buggy_consume_next(iterator)
buggy_consume_next(iterator)
print()

def buggy_consume_next(iterator):
    external_var.assign_add(next(iterator))
    tf.print("Value of external_var:", external_var)

print('Python iterator')
buggy_consume_next(iterator)
buggy_consume_next(iterator)
buggy_consume_next(iterator)
```
<br>
```code
tf.function iterator
Value of external_var: 0
Value of external_var: 0
Value of external_var: 0

Python iterator
Value of external_var: 1
Value of external_var: 3
Value of external_var: 6
```
<br>
iterator가 <code>tf.function</code>안에서 생성되고 수행되어지면 정확히 작동 될 것이다. 따라서 iterator를 구현하기 위해서 Function안에서 <code>x in y</code> 처럼 계속해서 반복하는 수 밖에 없고 그렇게 되면 **Large in-memory Dataset**이 되면서 감당할 수 없을 것이다.  
따라서 이러한 Iterator형식의 Dataset을 사용하기 위해서는 **<code>tf.data.Dataset.from_generator()</code>** 을 사용하여 Python data를 wrap해야 한다.  
아래 Code와 결과를 살펴보면 쉽게 이해 된다.  
간단한 train(), train2() <code>tf.function</code>은 dummy computation을 실시하게 된다.  
**Wrap하지 않은 연산자는 Data의 Size가 커지거나 연산이 추가될 수록 지속적으로 Node가 생성되면서 Graph가 커지는 반면 <code>tf.data.Dataset.from_generator()</code>Wrap을 실시할 시 Graph의 Size는 일정한 것을 확인할 수 있다.**  

**참조(tf.data.Dataset.from_generator)**  
>The generator argument must be a callable object that returns an object that supports the iter() protocol (e.g. a generator function). The elements generated by generator must be compatible with the given output_types and (optional) output_shapes arguments.

한 Object를 받아서 iter()을 적용할 수 있다.

참조: <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset?version=stable#__init__">tf.data 설명서</a>
```python
def measure_graph_size(f, *args):
    g = f.get_concrete_function(*args).graph
    print("{}({}) contains {} nodes in its graph".format(
        f.__name__, ', '.join(map(str, args)), 
        len(g.as_graph_def().node)))

@tf.function
def train(dataset):
    loss = tf.constant(0)
    for x, y in dataset:
        loss += tf.abs(y - x) # Some dummy computation.
    return loss

@tf.function
def train2(dataset):
    loss = tf.constant(0)
    for x, y in dataset:
        loss += tf.abs(y - x) # Some dummy computation.
        loss += tf.abs(y - tf.constant(0)) # Some dummy computation.
    return loss

small_data = [(1, 1)] * 2
big_data = [(1, 1)] * 10

print('No Wrap - train()')
measure_graph_size(train, small_data)
measure_graph_size(train, big_data)
print()

print('No Wrap - train2()')
measure_graph_size(train2, small_data)
measure_graph_size(train2, big_data)
print()

print('Wrap - train()')
measure_graph_size(train, tf.data.Dataset.from_generator(
    lambda: small_data, (tf.int32, tf.int32)))
measure_graph_size(train, tf.data.Dataset.from_generator(
    lambda: big_data, (tf.int32, tf.int32)))
print()

print('Wrap - train2()')
measure_graph_size(train2, tf.data.Dataset.from_generator(
    lambda: small_data, (tf.int32, tf.int32)))
measure_graph_size(train2, tf.data.Dataset.from_generator(
    lambda: big_data, (tf.int32, tf.int32)))
```
<br>
```code
No Wrap - train()
train([(1, 1), (1, 1)]) contains 8 nodes in its graph
train([(1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)]) contains 32 nodes in its graph

No Wrap - train2()
train2([(1, 1), (1, 1)]) contains 18 nodes in its graph
train2([(1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)]) contains 82 nodes in its graph

Wrap - train()
train(<DatasetV1Adapter shapes: (<unknown>, <unknown>), types: (tf.int32, tf.int32)>) contains 5 nodes in its graph
train(<DatasetV1Adapter shapes: (<unknown>, <unknown>), types: (tf.int32, tf.int32)>) contains 5 nodes in its graph

Wrap - train2()
train2(<DatasetV1Adapter shapes: (<unknown>, <unknown>), types: (tf.int32, tf.int32)>) contains 5 nodes in its graph
train2(<DatasetV1Adapter shapes: (<unknown>, <unknown>), types: (tf.int32, tf.int32)>) contains 5 nodes in its graph
```
<br>
<br><br>

### Automatic Control Dependencies
동일한 변수에 대하여 여러번 읽고 쓰는 것은 사용자가 원하는 flow대로 흐르지 않을 수 있다.  
tf.function을 사용하면 의도한 flow대로 자연스럽게 Code를 실행하는 것을 볼 수 있다.

```python
# Automatic control dependencies

a = tf.Variable(1.0)
b = tf.Variable(2.0)

@tf.function
def f(x, y):
    print('x: {}, y: {}'.format(x,y))
    a.assign(y * b) # 2*2
    b.assign_add(x * a) # 2 + 1*4
    tf.print('a: ',a,'b: ',b)
    return a + b # 4+6

f(1.0, 2.0)  # 10.0
```
<br>
x: 1.0, y: 2.0  
a:  4 b:  6  
<br><br>

### Variables
tf.function안에 tf.Variable을 생성할 경우 tf.function에서는 호출될 시 같은 variable로서 reuse되겠지만, eager mode에서는 호출될 시 각각 새로운 변수를 생성하게 될 것이다.  
tf.function은 사전에 이러한 Ambiguous한 구문을 사용하지 못하도록 막고 있다.

```python
# Ambiguous Code
print('Ambiguous Code')
@tf.function
def f(x):
    v = tf.Variable(1.0)
    v.assign_add(x)
    return v

with assert_raises(ValueError):
    f(1.0)
print()

# Non-ambiguous code
print('Non-ambiguous code')
v = tf.Variable(1.0)
@tf.function
def f(x):
    return v.assign_add(x)
print(f(1.0))
print(f(2.0))
print()

# 하나의 Object의 변수로서 tf.Variable 사용 가능
print('Object Variable')
class C:
    pass
obj = C()
obj.v = None

@tf.function
def g(x):
    if obj.v is None:
        obj.v = tf.Variable(1.0)
    return obj.v.assign_add(x)
print(g(1.0))
print(g(2.0))
```
<br>
```code
Ambiguous Code
Caught expected exception 
  <class 'ValueError'>:
  
Non-ambiguous code
tf.Tensor(2.0, shape=(), dtype=float32)
tf.Tensor(4.0, shape=(), dtype=float32)

Object Variable
tf.Tensor(2.0, shape=(), dtype=float32)
tf.Tensor(4.0, shape=(), dtype=float32)

...

    ValueError: tf.function-decorated function tried to create variables on non-first call.
```
<br>
<br><br>

### Using AutoGraph
AutoGraph와 관련된 라이브러리는 tf.function과 완전히 통일되어있다.  
Tensorflow의 Graph안에서의 반복이나 조건은 tf.wile_loop나 tf.cond로서 작성하는 것이 맞으나 이럴 경우 매우 복잡해지고 익숙한 제어와 조건문을 사용하길 원할 것 이다.  
**AutoGraph는 이러한 if같이 조건이나 for같이 반복문을 자동으로 Convert해서 Code를 실행한다.**  
<code>tf.autograph.to_code(function)</code>: 기존 Python Function을 tf.function으로 바꿀시 만드는 Code => tf.function이 어떻게 정의되어지는지 이해하기 편하라고 만들어둔 기능이나, 알아보기 매우 힘들다.

```python
# Simple loop

@tf.function
def f(x):
    while tf.reduce_sum(x) > 1:
        tf.print(x)
        x = tf.tanh(x)
    return x

f(tf.random.uniform([5]))

def f(x):
    while tf.reduce_sum(x) > 1:
        tf.print(x)
        x = tf.tanh(x)
    return x

print(tf.autograph.to_code(f))
```
<br>
```code
[0.520907283 0.82572329 0.908115149 0.760388494 0.711923599]
[0.478399932 0.678172946 0.720226347 0.641305745 0.611881673]
[0.44496125 0.590330303 0.617049456 0.565788 0.54545027]
[0.417748362 0.530133128 0.549070358 0.512259245 0.497102529]
[0.395031869 0.485482842 0.499823153 0.471703619 0.45983538]
[0.375690043 0.45062387 0.461978048 0.439574778 0.429950058]
[0.358958662 0.42241171 0.431695 0.413291931 0.405279577]
[0.344296455 0.398960233 0.40673691 0.39126426 0.38445729]
[0.331307679 0.379058957 0.385698527 0.372449636 0.366572112]
[0.319695294 0.361889958 0.367646068 0.356132507 0.350989729]
[0.309231371 0.346877664 0.351931036 0.341802895 0.337252975]
[0.299737662 0.333603591 0.338086963 0.329085976 0.325022757]
[0.291072518 0.321755022 0.325768441 0.317699224 0.314041436]
[0.283121645 0.311092943 0.314713418 0.307425052 0.304109246]
[0.275791764 0.30143106 0.304718971 0.298092753 0.295068622]
[0.269005805 0.292621672 0.29562515 0.289566249 0.286793262]
[0.262699485 0.284546 0.287303925 0.281735539 0.279180646]
[0.256818712 0.277107239 0.279651463 0.274510592 0.272146583]
[0.251317561 0.270225644 0.272582471 0.267816931 0.265621096]
[0.246156782 0.263834774 0.266026169 0.261592329 0.259545565]
[0.241302595 0.25787881 0.259923309 0.255784273 0.253870428]
[0.236725733 0.252310455 0.254223794 0.25034821 0.248553455]
[0.232400686 0.247089297 0.248884961 0.245245948 0.243558407]
[0.228305146 0.242180616 0.243870214 0.240444601 0.238853991]
[0.224419475 0.237554371 0.239147976 0.235915646 0.234413]
[0.220726296 0.233184427 0.234690815 0.231634215 0.23021169]
[0.217210203 0.229047909 0.23047477 0.227578506 0.226229221]
[0.213857457 0.225124717 0.226478815 0.223729312 0.222447187]
[0.210655764 0.221397072 0.222684413 0.220069662 0.218849286]
[0.207594097 0.21784924 0.219075143 0.216584459 0.215421021]
[0.204662517 0.214467183 0.215636387 0.213260248 0.212149441]
[0.201852053 0.211238354 0.212355107 0.21008499 0.209022954]
[0.199154571 0.208151504 0.20921962 0.20704785 0.206031114]
[0.196562693 0.205196515 0.206219435 0.204139084 0.203164518]
[0.194069698 0.202364236 0.20334506 0.201349899 0.200414658]
def tf__f(x):
  do_return = False
  retval_ = ag__.UndefinedReturnValue()
  with ag__.FunctionScope('f', 'f_scope', ag__.ConversionOptions(recursive=True, user_requested=True, optional_features=(), internal_convert_user_code=True)) as f_scope:

    def get_state():
      return ()

    def set_state(_):
      pass

    def loop_body(x):
      ag__.converted_call(tf.print, f_scope.callopts, (x,), None, f_scope)
      x = ag__.converted_call(tf.tanh, f_scope.callopts, (x,), None, f_scope)
      return x,

    def loop_test(x):
      return ag__.converted_call(tf.reduce_sum, f_scope.callopts, (x,), None, f_scope) > 1
    x, = ag__.while_stmt(loop_test, loop_body, get_state, set_state, (x,), ('x',), ())
    do_return = True
    retval_ = f_scope.mark_return_value(x)
  do_return,
  return ag__.retval(retval_)
```
<br>
<br><br>

#### AutoGraph: Conditionals
<a href="https://www.tensorflow.org/api_docs/python/tf/cond?version=stable">tf.cond</a>를 살펴보게 되면 **Condition을 비교하여 각각의 Function을 적용할 수 있다.**  
아래 tf.cond() Example을 살펴보게 되면 x < y 이기 때문에 True가 되고 따라서 f1이 실행됨에 따라서 2 * 17의 결과인 37을 출력하게 된다.  

또한 Tensor로 Casting된 True는 tf.cond로서 비교하게 된다.

```python
# tf.cond() Example
print('tf.cond() Example')
x = tf.constant(2)
y = tf.constant(5)
def f1(): return tf.multiply(x, 17)
def f2(): return tf.add(y, 23)
r = tf.cond(tf.less(x, y), f1, f2)
print(r.numpy())
print()

# Python IF 문인지 tf.cond인지 판단하는 Function
def test_tf_cond(f, *args):
    g = f.get_concrete_function(*args).graph
    if any(node.name == 'cond' for node in g.as_graph_def().node):
        print("{}({}) uses tf.cond.".format(
            f.__name__, ', '.join(map(str, args))))
    else:
        print("{}({}) executes normally.".format(
            f.__name__, ', '.join(map(str, args))))

    print("  result: ",f(*args).numpy())
    
@tf.function
def dropout(x, training=True):
    if training:
        x = tf.nn.dropout(x, rate=0.5)
    return x

print('if vs tf.cond')
print('Use Parameter')
test_tf_cond(dropout, tf.ones([10], dtype=tf.float32), True)
print()
print('Use Tensor(tf.constant(True))')
test_tf_cond(dropout, tf.ones([10], dtype=tf.float32), tf.constant(True))
```
<br>
```code
tf.cond() Example
34

if vs tf.cond
Use Parameter
dropout(tf.Tensor([1. 1. 1. 1. 1. 1. 1. 1. 1. 1.], shape=(10,), dtype=float32), True) executes normally.
  result:  [2. 2. 0. 2. 0. 2. 2. 2. 0. 0.]

Use Tensor(tf.constant(True))
dropout(tf.Tensor([1. 1. 1. 1. 1. 1. 1. 1. 1. 1.], shape=(10,), dtype=float32), tf.Tensor(True, shape=(), dtype=bool)) uses tf.cond.
  result:  [0. 2. 0. 2. 0. 0. 2. 0. 0. 0.]
```
<br>
<span style="color:red">tf.function</span>에서 매우 중요한 부분이다.  

**<code>tf.function</code>에서 비교하는 값이 Tensor이게 되면 if나 else와 같은 조건문을 tf.cond로서 판별하게 된다.**  

**중요한점은 이러한 tf.cond는 일반적인 Python처럼 하나의 조건(if or else)만 살펴보는 것이 아니라 양쪽 다 살펴본 뒤 하나를 선택하게 된다.**  

<span style="color:red">**따라서 tf.constant(True)와 같이 Tensor로 감싼 True를 조건문으로서 판단하게 되면 꼭 if, else를 같이 사용하여야 한다.**</span>  

**마지막으로 Tensor로 감싼 True는 Python에서 True와 다르기 때문에 bool()같은 Type으로서 Casting시 TypeError가 발생한다.**  

기본적인 Python 문법과 같다고 생각하고 Code를 작성하게 되면 값은 일정하나 Code안에서 print()와 같이 즉시 시행되는 것에서 예상치 못하는 Error가 발생할 수 있다.  
최종적으로 정리하면 다음과 같다.
- 비교하는 값이 Tensor이게 되면 if나 else와 같은 조건문을 tf.cond로서 판별하게 된다.
- tf.cond는 일반적인 Python처럼 하나의 조건(if or else)만 살펴보는 것이 아니라 양쪽 다 살펴본 뒤 하나를 선택하게 된다.
- Tensor로 감싼 True는 Python에서 True와 다르기 때문에 bool()같은 Type으로서 Casting시 TypeError가 발생한다.

```python
@tf.function
def f(x):
    if x > 0:
        x = x + 1.
        print("Tracing `then` branch")
    else:
        x = x - 1.
        print("Tracing `else` branch")
    return x

print('Use Python Parameter')
print(f(-1.0).numpy())
print(f(1.0).numpy())
print()

print('Use Tensor Parameter')
print(f(tf.constant(1.0)).numpy())
print()

@tf.function
def f():
    x = None
    if tf.constant(True):
        x = tf.ones([3, 3])
    else:
        x = tf.ones([2,2])
    return x

result = f()
print('Use If & Else')
print(result.numpy())
print()
    
@tf.function
def f():
    if tf.constant(True):
        x = tf.ones([3, 3])
    return x

# Throws an error because both branches need to define `x`.
print('Use only If')
with assert_raises(ValueError):
    f()
print()

# Bool Casting
@tf.function
def f(x, y):
    if bool(x):
        y = y + 1.
        print("Tracing `then` branch")
    else:
        y = y - 1.
        print("Tracing `else` branch")
    return y

print('Bool Casting')
print('Using True or False')
print(f(True, 0).numpy())
print(f(False, 0).numpy())
print()

print('Using tf.constant(True)')
with assert_raises(TypeError):
    f(tf.constant(True), 0.0)
```
<br>
```code
Use Python Parameter
Tracing `else` branch
-2.0
Tracing `then` branch
2.0

Use Tensor Parameter
Tracing `then` branch
Tracing `else` branch
2.0

Use If & Else
[[1. 1. 1.]
 [1. 1. 1.]
 [1. 1. 1.]]

Use only If
Caught expected exception 
  <class 'ValueError'>:

Bool Casting
Using True or False
Tracing `then` branch
1.0
Tracing `else` branch
-1.0

Using tf.constant(True)
Caught expected exception 
  <class 'TypeError'>:
  
...

    ValueError: The following symbols must also be initialized in the else branch: ('x',). Alternatively, you may initialize them before the if statement.

...

    OperatorNotAllowedInGraphError: using a `tf.Tensor` as a Python `bool` is not allowed: AutoGraph did not convert this function. Try decorating it directly with @tf.function.
```
<br>
<br><br>

#### AutoGraph: Loop
AutoGraph는 Loop구문을 다음과 같이 Convert한다.
- for: Convert if the iterable is a tensor
- while: Convert if the while condition depends on a tensor

위의 Convert를 자세히 알아보면 다음과 같다.  
- 반복문의 범위가 Tensor로 주어진다. -> <a href="https://www.tensorflow.org/api_docs/python/tf/while_loop?version=stable">tf.while_loop</a>사용
- for x in tf.data.Dataset 으로 주어딘다. -> <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset?version=stable#reduce">tf.data.Dataset,redyce</a>사용

```python
# AutoGraph에서 Convert하는 Loop구문 확인
def test_dynamically_unrolled(f,*args):
    g = f.get_concrete_function(*args).graph
    if any(node.name == 'while' for node in g.as_graph_def().node):
        # 반복문의 범위가 Tensor로서 주어짐
        print("{}({}) uses tf.while_loop.".format(
            f.__name__, ', '.join(map(str, args))))
    elif any(node.name == 'ReduceDataset' for node in g.as_graph_def().node):
        # for x in tf.data.Dataset 
        print("{}({}) uses tf.data.Dataset.reduce.".format(
            f.__name__, ', '.join(map(str, args))))
    else:
        # 기본적인 반복문 사용
        print("{}({}) gets unrolled.".format(
            f.__name__, ', '.join(map(str, args))))
    print()
        
# 기본적인 반복문 사용
@tf.function
def for_in_range():
    x = 0
    for i in range(5):
        x += i
    return x
print('기본적인 반복문 사용')
test_dynamically_unrolled(for_in_range)

# 반복문의 범위가 Tensor로서 사용
@tf.function
def for_in_tfrange():
    x = tf.constant(0, dtype=tf.int32)
    for i in tf.range(5):
        x += i
    return x
print('반복문의 범위가 Tensor로서 사용')
test_dynamically_unrolled(for_in_tfrange)

# for x in tf.data.Dataset 형태
@tf.function
def for_in_tfdataset():
    x = tf.constant(0, dtype=tf.int64)
    for i in tf.data.Dataset.range(5):
        x += i
    return x
print('for x in tf.data.Dataset 형태')
test_dynamically_unrolled(for_in_tfdataset)

# 반복문 + 조건문
@tf.function
def while_py_cond():
    x = 5
    while x > 0:
        x -= 1
    return x
print('반복문 + 조건문')
test_dynamically_unrolled(while_py_cond)

# 반복문(범위 Tensor) + 조건문(변수 Tensor)
@tf.function
def while_tf_cond():
    x = tf.constant(5)
    while x > 0:
        x -= 1
    return x
print('반복문(범위 Tensor) + 조건문(변수 Tensor)')
test_dynamically_unrolled(while_tf_cond)
```
<br>
```code
기본적인 반복문 사용
for_in_range() gets unrolled.

반복문의 범위가 Tensor로서 사용
for_in_tfrange() uses tf.while_loop.

for x in tf.data.Dataset 형태
for_in_tfdataset() uses tf.data.Dataset.reduce.

반복문 + 조건문
while_py_cond() gets unrolled.

반복문(범위 Tensor) + 조건문(변수 Tensor)
while_tf_cond() uses tf.while_loop.
```
<br>

<hr>
참조: <a href="https://github.com/wjddyd66/Tensorflow2.0/blob/master/tf.function.ipynb">원본코드</a><br>
참조: <a href="https://www.tensorflow.org/tutorials/customization/performance">tf.function</a><br>
코드에 문제가 있거나 궁금한 점이 있으면 wjddyd66@naver.com으로  Mail을 남겨주세요.

