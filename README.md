# 제어공학1 5주차 3장과제
## 2021732030 김준호
***

### Problems 3.1
(a) 적당한 상태변수를 설정하여라. 
(b) 상태변수로 표현된 1차 미분방정식을 구시오.  
(c) 상태미분방정식을 구하시오.

*풀이과정*

$$
F(t) = M \frac{d^2 y(t)}{dt^2} + b \frac{d y(t)}{dt} + ky(t)
$$

이때 상태변수를 다음과 같이 정의한다.

$$
x_1(t) = y(t), \quad x_2(t) = \dot{y}(t)
$$

따라서,

$$
y(t) = x_1(t)
$$

$$
\frac{d}{dt} y(t) = x_2(t) = \frac{d}{dt} x_1(t)
$$

$$
\frac{d^2}{dt^2} y(t) = \frac{d}{dt} x_2(t)
$$
상태변수를 구햇으므로 1차 미분방정식을 구하는 방법은 다음과 같다. 
$$
\dot{x}_1(t) = x_2(t)
$$

$$
\dot{x}_1(t) = x_2(t)
$$

$$
\dot{x}_2(t) = -\frac{k}{M} x_1(t) - \frac{b}{M} x_2(t) + \frac{1}{M} F(t)
$$

상태미분방정식은

$$
\dot{X}(t) = A X(t) + B F(t)
$$

으로 구할 수 있다. 따라서

$$
X(t) =
\begin{bmatrix}
x_1(t) \\
x_2(t)
\end{bmatrix}
$$

$$
\dot{X}(t) =
\begin{bmatrix}
0 & 1 \\
-\dfrac{k}{M} & -\dfrac{b}{M}
\end{bmatrix}
X(t)
+
\begin{bmatrix}
0 \\
\dfrac{1}{M}
\end{bmatrix}
F(t)
$$

***
### Problems 3.3
그림 P3.3과 같은 RLC 회로가 주어졌다. 상태변수  
$(x_1(t) = i_L(t) ), ( x_2(t) = v_C(t))$ 로 설정하고 상태미분방정식을 구하라.

부분해답:

$$
A =
\begin{bmatrix}
0 & \dfrac{1}{L} \\
-\dfrac{1}{C} & -\dfrac{1}{RC}
\end{bmatrix}
$$

<ㅇㅇ>

상태방정식 유도 (RLC 회로)

상태변수 정의를 해보면

$$
x_1(t) = i_L(t), \quad x_2(t) = v_C(t)
$$

Node1 에서 KCL 적용하였을 때 다음과 같은식을 세울 수 있다. 

$$
x_1(t) = \frac{v_2(t) - x_2(t)}{R} - C \frac{d}{dt} x_2(t)
$$

Loop에서 KVL 적용해보면 

$$
v_1(t) = L \frac{d}{dt} x_1(t) + v_2(t) - x_2(t)
$$


상태변수로 표현한 1차 미분방정식은 다음과 같다. 

$$
\frac{d}{dt} x_1(t) = \frac{1}{L} x_2(t) + \frac{1}{L} v_1(t) - \frac{1}{L} v_2(t)
$$

$$
\frac{d}{dt} x_2(t) = -\frac{1}{C} x_1(t) - \frac{1}{RC} x_2(t) + \frac{1}{RC} v_2(t)
$$

상태미분방정식을 행렬형식으로 바꾸면 다음과 같다. 

$$
\dot{X}(t) = A X(t) + B V(t)
$$

$$
\dot{X}(t) =
\begin{bmatrix}
0 & \dfrac{1}{L} 
-\dfrac{1}{C} & -\dfrac{1}{RC}
\end{bmatrix}
X(t)
+
\begin{bmatrix}
\dfrac{1}{L} & -\dfrac{1}{L} 
0 & \dfrac{1}{RC}
\end{bmatrix}
V(t)
$$

***

### Problems 3.5
그림 P3.5에 폐루프 제어시스템이 주어져 있다.  
(a) 폐루프 전달함수 $( T(s) = \dfrac{Y(s)}{R(s)} )$를 구하라.  
(b) 상태변수 모델을 구하고 위상변수형 블록선도를 작성하라.

<ㅇㅇ>

(a) 폐루프 전달함수는 다음과 같다:

$$
G(s) = \frac{s + 2}{s + 8} \cdot \frac{1}{s - 3} \cdot \frac{1}{s}
$$

이를 정리하면

$$
G(s) = \frac{s + 2}{s (s - 3)(s + 8)}
$$

폐루프 전달함수 \( T(s) \)는 다음과 같이 표현된다:

$$
T(s) = \frac{G(s)}{1 + G(s)}
$$

따라서

$$
T(s) = \frac{\dfrac{s + 2}{s (s - 3)(s + 8)}}{1 + \dfrac{s + 2}{s (s - 3)(s + 8)}}
$$

이를 정리하여 나타내면 

$$
T(s) = \frac{s + 2}{s^3 + 5s^2 - 23s + 2}
$$

(b) 상태변수 모델 유도하기 위해 우선 전달함수를 이용해보면

$$
T(s) = \frac{(s + 2) Z(s)}{(s^3 + 5s^2 - 23s + 2)Z(s)}
$$

$$Y(s)=sZ(s)+2Z(s)$$

$$R(s)=s^3Z(s)+5s^2Z(s)-23sZ(s)+2Z(s)$$

이를 시간영역으로 역변환하여주면

$$
y(t)= \dot{z}(t) + 2 z(t) 
$$

$$
r(t)=\dddot{z}(t) + 5\ddot{z}(t) - 23\dot{z}(t) + 2 z(t) 
$$

따라서 상태변수를 다음과 같이 정의한다:

$$
x_1(t) = z(t), \quad x_2(t) = \dot{z}(t), \quad x_3(t) = \ddot{z}(t)
$$

이를 이용하면 상태방정식은 다음과 같다. 

$$
\begin{aligned}
\dot{x}_1(t) &= x_2(t) \\
\dot{x}_2(t) &= x_3(t) \\
\dot{x}_3(t) &= -5x_3(t) + 23x_2(t) - 2x_1(t) + u(t)
\end{aligned}
$$

출력 방정식은 다음과 같다

$$
y(t) = 2x_1(t) + x_2(t)
$$

(c) 상태방정식을 행렬형태로 표현하면

$$
\dot{X}(t) =
\begin{bmatrix}
0 & 1 & 0 \\
0 & 0 & 1 \\
-2 & 23 & -5
\end{bmatrix}
X(t)
+
\begin{bmatrix}
0 \\
0 \\
1
\end{bmatrix}
u(t)
$$

이고, 출력방정식은 

$$
y(t) =
\begin{bmatrix}
2 & 1 & 0
\end{bmatrix}
X(t)
$$

***

### Problems 3.12

전달함수가 다음과 같이 주어진 시스템에서
(a) 상태공간모델을 구하라.  
(b) 상태천이행렬 $( \Phi(t) )$를 구하라.

$$
\frac{Y(s)}{R(s)} = T(s) = \frac{8(s + 5)}{s^3 + 12s^2 + 44s + 48}
$$

(a) 전달함수는 다음과 같다

$$
T(s) = \frac{8(s + 5)Z(s)}{(s^3 + 12s^2 + 44s + 48)Z(s)}
$$

따라서

$$
Y(s) = 8 (s + 5) Z(s) = 8s Z(s) + 40 Z(s)
$$

또한 입력 \( R(s) \)는

$$
R(s) = (s^3 + 12s^2 + 44s + 48) Z(s)=s^3Z(s)+12s^2Z(s)+44sZ(s)+48Z(s)
$$

이고, 이를  라플라스 역변환을 적용해보면

$$
y(t) = 8 \dot{z}(t) + 40 z(t)
$$

$$
r(t) = \dddot{z}(t) + 12 \ddot{z}(t) + 44 \dot{z}(t) + 48 z(t)
$$

이다. 상태변수 정의 다음과 같다따라서  정리해보면 

$$
x_1(t) = z(t), \quad x_2(t) = \dot{z}(t), \quad x_3(t) = \ddot{z}(t)
$$

$$
\dot{x}_1(t) = x_2(t)
$$

$$
\dot{x}_2(t) = x_3(t)
$$

$$
\dot{x}_3(t) = -48 x_1(t) - 44 x_2(t) - 12 x_3(t) + r(t)
$$

따라서 출력방정식은 다음과 같다:

$$
y(t) = 40 x_1(t) + 8 x_2(t)
$$

상태방정식을 행렬형태로 표현해보면 

$$
\dot{X}(t) = A X(t) + B R(t)
$$

$$
Y(t) = C X(t) + D R(t)
$$


$$
A =
\begin{bmatrix}
0 & 1 & 0 \\
0 & 0 & 1 \\
-48 & -44 & -12
\end{bmatrix},
\quad
B =
\begin{bmatrix}
0 \\
0 \\
1
\end{bmatrix},
\quad
C =
\begin{bmatrix}
40 & 8 & 0
\end{bmatrix},
\quad
D = [0]
$$

(b) 상태천이행렬 $( \Phi(s) )$ 은 다음과 같이 정의된다:

$$
\Phi(s) = (sI - A)^{-1}
$$
이때 $\Phi(s) = (sI - A)^{-1}=\frac{\text{adj}(sI - A)}{|sI - A|}$

$$
sI - A =
\begin{bmatrix}
s & -1 & 0 \\
0 & s & -1 \\
48 & 44 & s + 12
\end{bmatrix}
$$

이므로 행렬식은 다음과 같다

$$
|sI - A| = s^3 + 12 s^2 + 44 s + 48
$$

여기서 adjoint를 구해보면 

$$
\text{adj}(sI - A) =
\begin{bmatrix}
s^2 + 12s + 44 & s + 12 & 1 \\
-48 & s^2 & s \\
-48 & -48 & s^2
\end{bmatrix}
$$

따라서 상태천이행렬은 다음과 같다.

$$
\Phi(s) = \frac{1}{s^3 + 12 s^2 + 44 s + 48}
\begin{bmatrix}
s^2 + 12s + 44 & s + 12 & 1 \\
-48 & s^2 & s \\
-48 & -48 & s^2
\end{bmatrix}
$$

이때 천이행렬의 라플라스 역변환을 통해 $\Phi(t)$를 MATLAB을 사용하여 라플라스 역변환을 구해보면 다음과 같다. 


$$
\Phi(t) =
\begin{bmatrix}
3e^{-2t} - 3e^{-4t} + e^{-6t} &
5e^{-2t} - 2e^{-4t} + \dfrac{3}{4}e^{-6t} &
\dfrac{1}{8}e^{-2t} - \dfrac{1}{4}e^{-4t} + \dfrac{1}{8}e^{-6t} \\
-6e^{-2t} + 12e^{-4t} - 6e^{-6t} &
-\dfrac{5}{2}e^{-2t} + 8e^{-4t} + \dfrac{9}{2}e^{-6t} &
-\dfrac{1}{4}e^{-2t} + e^{-4t} + \dfrac{3}{4}e^{-6t} \\
12e^{-2t} - 48e^{-4t} + 36e^{-6t} &
5e^{-2t} - 32e^{-4t} + 27e^{-6t} &
\dfrac{1}{2}e^{-2t} - 4e^{-4t} + \dfrac{9}{2}e^{-6t}
\end{bmatrix}
$$



***

### Problems 3.17

다음과 같은 상태변수 방정식으로 표현된 시스템이 있다.

$$
\dot{X}(t) =
\begin{bmatrix}
1 & 1 & -1 \\
4 & 3 & 0 \\
-2 & 1 & 10
\end{bmatrix} X(t) +
\begin{bmatrix}
0 \\
0 \\
4
\end{bmatrix} u(t)
$$

$$
y(t) =
\begin{bmatrix}
1 & 0 & 0
\end{bmatrix} X(t)
$$

$$
G(s) = \frac{Y(s)}{U(s)} \text{ 를 구하라.}
$$

주어진 상태공간은 다음과 같고,  

$$
\dot{X}(t)=AX(t)+Bu(t),\qquad y(t)=CX(t)
$$

$$
A=\begin{bmatrix}
1 & 1 & -1 \\
4 & 3 & 0 \\
-2 & 1 & 10
\end{bmatrix},\quad
B=\begin{bmatrix}0 \\ 0 \\ 4\end{bmatrix},\quad
C=\begin{bmatrix}1 & 0 & 0\end{bmatrix},\quad
D=0
$$

전달함수는 다음과 같다.  

$$
G(s)=C\,(sI-A)^{-1}B + D
$$

이때 $(sI - A)^{-1}=\frac{\text{adj}(sI - A)}{|sI - A|}$

$$
sI - A =
\begin{bmatrix}
s - 1 & -1 & 1 \\
-4 & s - 3 & 0 \\
2 & -1 & s - 10
\end{bmatrix}
$$

$$
|sI - A| = s^3 - 14s^2 + 37s + 20
$$

$\text{adj}(sI - A)$를 구해보면

$$
\mathrm{adj}(sI - A) =
\left[
\begin{array}{ccc}
s^2 - 13s + 30 & 4s - 40 & -2s + 10 \\
s - 11 & s^2 - 11s + 8 & s - 3 \\
-s + 3 & -4 & s^2 - 4s - 1
\end{array}
\right]^T
$$


따라서 $(sI - A)^{-1}$ 은 다음과 같다.

$$
(sI - A)^{-1} =
\frac{1}{s^3 - 14s^2 + 37s + 20}
\begin{bmatrix}
s^2 - 13s + 30 & s - 11 & -s + 3 \\
4s - 40 & s^2 - 11s + 8 & -4 \\
-2s + 10 & s - 3 & s^2 - 4s - 1
\end{bmatrix}
$$

즉 상태천이행렬 $\Phi(s)$ 는 다음과 같다

$$
\Phi(s) =
\frac{1}{s^3 - 14s^2 + 37s + 20}
\begin{bmatrix}
s^2 - 13s + 30 & s - 11 & -s + 3 \\
4s - 40 & s^2 - 11s + 8 & -4 \\
-2s + 10 & s - 3 & s^2 - 4s - 1
\end{bmatrix}
$$

전달함수는

$$
G(s) = C \Phi(s) B + D
$$

이므로 여기서 다음 식을 계산해보면   

$$
C = [1 \; 0 \; 0],  
B = \begin{bmatrix} 0 \\ 0 \\ 4 \end{bmatrix},  
D = 0
$$

$$
G(s) = \frac{-4s + 12}{s^3 - 14s^2 + 37s + 20}
$$

***
