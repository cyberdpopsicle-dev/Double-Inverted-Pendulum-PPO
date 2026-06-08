# Double Inverted Pendulum Dynamics

## State Vector

The system state is defined as

[
x =
\begin{bmatrix}
\phi_1 \
\dot{\phi}_1 \
\phi_2 \
\dot{\phi}_2
\end{bmatrix}
]

where

* (\phi_1) : first link angle
* (\dot{\phi}_1) : first link angular velocity
* (\phi_2) : second link angle
* (\dot{\phi}_2) : second link angular velocity

---

## Friction Torques

Viscous damping is modeled as

[
M_{R1}=R_{p1}\dot{\phi}_1
]

[
M_{R2}=R_{p2}\left(\dot{\phi}_2-\dot{\phi}_1\right)
]

---

## Equations of Motion

The nonlinear dynamics are expressed in matrix form as

[
H(q)\ddot{q}=F(q,\dot q,u)
]

with

[
q=
\begin{bmatrix}
\phi_1 \
\phi_2
\end{bmatrix}
]

and

[
H=
\begin{bmatrix}
a_{11} &
a_{12}\cos(\phi_1-\phi_2)
\
a_{12}\cos(\phi_1-\phi_2) &
a_{21}
\end{bmatrix}
]

---

## Generalized Forces

[
F=
\begin{bmatrix}
F_1 \
F_2
\end{bmatrix}
]

where

[
F_1=
-a_{12}\sin(\phi_1-\phi_2)\dot{\phi}*2^{,2}
-a*{13}\sin(\phi_1)
+u
-M_{R1}
]

[
F_2=
a_{12}\sin(\phi_1-\phi_2)\dot{\phi}*1^{,2}
-a*{22}\sin(\phi_2)
-M_{R2}
]

and (u) is the motor torque input.

---

## Dynamic Inversion

The angular accelerations are obtained from

[
\ddot q = H^{-1}F
]

Expanding explicitly,

[
\ddot{\phi}*1=
\frac{H*{22}F_1-H_{12}F_2}
{\det(H)}
]

[
\ddot{\phi}*2=
\frac{-H*{21}F_1+H_{11}F_2}
{\det(H)}
]

with

[
\det(H)=H_{11}H_{22}-H_{12}H_{21}
]

---

## State-Space Representation

The continuous-time system is represented as

[
\dot{x}
=======

\begin{bmatrix}
\dot{\phi}_1 \
\ddot{\phi}_1 \
\dot{\phi}_2 \
\ddot{\phi}_2
\end{bmatrix}
]

---

## Numerical Integration

State propagation is performed using a fourth-order Runge–Kutta (RK4) integrator.

[
k_1=f(x_n)
]

[
k_2=f\left(x_n+\frac{h}{2}k_1\right)
]

[
k_3=f\left(x_n+\frac{h}{2}k_2\right)
]

[
k_4=f\left(x_n+h,k_3\right)
]

[
x_{n+1}
=======

x_n
+
\frac{h}{6}
\left(
k_1+2k_2+2k_3+k_4
\right)
]

where

[
h = 0.002\ \text{s}
]

---

## PPO Control Law

The reinforcement learning policy generates a continuous torque command

[
u=\pi_\theta(x)
]

with

[
u\in[-6,6]\ \text{Nm}
]

using the observed state

[
x=
\begin{bmatrix}
\phi_1 &
\dot{\phi}_1 &
\phi_2 &
\dot{\phi}_2
\end{bmatrix}
]

The control objective is stabilization of the upright equilibrium

[
(\phi_1,\phi_2)=(\pi,\pi)
]

for both pendulum links.
