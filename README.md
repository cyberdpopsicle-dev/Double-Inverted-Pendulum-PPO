# Double Inverted Pendulum Reinforcement Learning Controller

## Abstract

Maintaining balance in highly unstable robotic systems remains a fundamental challenge in modern control engineering. This project investigates the application of Deep Reinforcement Learning (DRL) to the control of a wheel-based stair-climbing robotic mechanism modeled as a Double Inverted Pendulum (DIP). A physics-based simulation environment was developed using non-linear equations of motion derived from Lagrangian mechanics and integrated using a fourth-order Runge-Kutta numerical solver.

A Proximal Policy Optimization (PPO) agent was trained to generate continuous motor torque commands that stabilize the system around an upright equilibrium configuration. To improve model transparency and interpretability, Explainable Artificial Intelligence (XAI) techniques including SHAP and LIME were integrated into an interactive Streamlit dashboard. The resulting platform enables real-time visualization of system states, controller outputs, and feature contribution analysis.

---

## Project Objectives

The primary objectives of this capstone project are:

* Develop a mathematical model of a double inverted pendulum system.
* Implement a high-fidelity physics simulation environment.
* Create a Gymnasium-compatible reinforcement learning environment.
* Train a PPO-based controller using Stable-Baselines3.
* Evaluate system stabilization performance under varying initial conditions.
* Integrate Explainable AI methods to interpret controller decisions.
* Design an interactive dashboard for visualization and analysis.

---

## System Architecture

The project consists of four major subsystems:

### 1. Physics Simulation Engine

The simulation engine models the non-linear dynamics of a double inverted pendulum using:

* Lagrangian Mechanics
* Forward Dynamic Inversion
* Runge-Kutta Fourth-Order Integration (RK4)

State Vector:

x = [φ₁, φ̇₁, φ₂, φ̇₂]

where:

* φ₁ = Chassis angle
* φ̇₁ = Chassis angular velocity
* φ₂ = Wheel-link angle
* φ̇₂ = Wheel-link angular velocity

---

### 2. Reinforcement Learning Environment

A custom Gymnasium environment was developed to interface the physical simulator with reinforcement learning algorithms.

#### Observation Space

[φ₁, φ̇₁, φ₂, φ̇₂]

#### Action Space

u ∈ [-6 Nm, +6 Nm]

where u represents the motor torque applied to the wheel mechanism.

#### Reward Function

The reward function combines:

* Angular stabilization error
* Angular velocity minimization
* Control effort minimization

to encourage efficient and stable balancing behavior.

---

### 3. PPO Controller

The controller is trained using Proximal Policy Optimization (PPO), a policy-gradient reinforcement learning algorithm known for:

* Stable convergence
* Continuous control capabilities
* Sample efficiency
* Robust performance in non-linear systems

Training is performed using Stable-Baselines3 with a multilayer perceptron policy network.

---

### 4. Explainable AI Dashboard

The Streamlit dashboard provides:

* Real-time state visualization
* Torque prediction monitoring
* Forward trajectory simulation
* SHAP feature attribution analysis
* LIME local explanation analysis
* Interactive Plotly telemetry charts

This allows users to understand not only what actions the controller takes, but also why those actions are selected.

---

## Technologies Used

### Programming Languages

* Python

### Machine Learning

* Stable-Baselines3
* PyTorch
* PPO

### Simulation

* NumPy
* SciPy
* Gymnasium

### Explainable AI

* SHAP
* LIME

### Visualization

* Streamlit
* Plotly
* Pandas

---

## Repository Structure

double-inverted-pendulum-ppo/

├── app.py

├── train_ppo.py

├── double_pendulum_env.py

├── double_pendulum_gym.py

├── requirements.txt

├── README.md

├── LICENSE

├── images/

├── models/

└── docs/

---

## Installation

Clone the repository:

git clone https://github.com/YOUR_USERNAME/double-inverted-pendulum-ppo.git

cd double-inverted-pendulum-ppo

Install dependencies:

pip install -r requirements.txt

---

## Training the PPO Agent

python train_ppo.py

After training completes, the model is saved as:

ppo_double_pendulum_model.zip

---

## Running the Dashboard

streamlit run app.py

The dashboard will launch locally and provide real-time monitoring and explainability features.

---

## Results

Expected outcomes include:

* Successful stabilization of the double inverted pendulum around the upright equilibrium point.
* Reduced angular deviations and oscillations.
* Learned control policies capable of continuous balancing.
* Explainability visualizations highlighting influential state variables in controller decision-making.

---

## Future Work

Potential future improvements include:

* Hardware deployment on a physical robotic prototype.
* Sim-to-real transfer learning.
* Multi-agent reinforcement learning approaches.
* Comparison with classical controllers such as PID, LQR, and MPC.
* Advanced XAI methods for policy interpretation.
* Integration of computer vision and sensor fusion modules.

---

## Author

Tehlil

Bachelor of Science Capstone Project

Department of Engineering and Computer Science

2026
