# Double Inverted Pendulum Dynamics

## State Vector

\[
x =
\begin{bmatrix}
\phi_1 \\
\dot{\phi}_1 \\
\phi_2 \\
\dot{\phi}_2
\end{bmatrix}
\]

---

## Friction Torques

\[
M_{R1}=R_{p1}\dot{\phi}_1
\]

\[
M_{R2}=R_{p2}\left(\dot{\phi}_2-\dot{\phi}_1\right)
\]

---

## Equations of Motion

\[
H(q)\ddot{q}=F(q,\dot{q},u)
\]

where

\[
q=
\begin{bmatrix}
\phi_1 \\
\phi_2
\end{bmatrix}
\]

and

\[
H=
\begin{bmatrix}
a_{11} &
a_{12}\cos(\phi_1-\phi_2)
\\
a_{12}\cos(\phi_1-\phi_2) &
a_{21}
\end{bmatrix}
\]

---

## Nonlinear Force Vector

\[
F=
\begin{bmatrix}
F_1 \\
F_2
\end{bmatrix}
\]

\[
F_1=
-a_{12}\sin(\phi_1-\phi_2)\dot{\phi}_2^{\,2}
-a_{13}\sin(\phi_1)
+u
-M_{R1}
\]

\[
F_2=
a_{12}\sin(\phi_1-\phi_2)\dot{\phi}_1^{\,2}
-a_{22}\sin(\phi_2)
-M_{R2}
\]

---

## Forward Dynamics

\[
\ddot{q}=H^{-1}F
\]

\[
\det(H)=H_{11}H_{22}-H_{12}H_{21}
\]

\[
\ddot{\phi}_1=
\frac{H_{22}F_1-H_{12}F_2}
{\det(H)}
\]

\[
\ddot{\phi}_2=
\frac{-H_{21}F_1+H_{11}F_2}
{\det(H)}
\]

---

## State-Space Form

\[
\dot{x}
=
\begin{bmatrix}
\dot{\phi}_1 \\
\ddot{\phi}_1 \\
\dot{\phi}_2 \\
\ddot{\phi}_2
\end{bmatrix}
\]

---

## Runge–Kutta 4 Integration

\[
k_1=f(x_n)
\]

\[
k_2=f\left(x_n+\frac{h}{2}k_1\right)
\]

\[
k_3=f\left(x_n+\frac{h}{2}k_2\right)
\]

\[
k_4=f\left(x_n+hk_3\right)
\]

\[
x_{n+1}
=
x_n
+
\frac{h}{6}
\left(
k_1+2k_2+2k_3+k_4
\right)
\]

\[
h=0.002\ \text{s}
\]

---

## PPO Control Law

\[
u=\pi_\theta(x)
\]

\[
x=
\begin{bmatrix}
\phi_1 &
\dot{\phi}_1 &
\phi_2 &
\dot{\phi}_2
\end{bmatrix}^{T}
\]

\[
u\in[-6,6]\ \text{Nm}
\]

---

## Upright Equilibrium

\[
(\phi_1,\phi_2)
=
(\pi,\pi)
\]
