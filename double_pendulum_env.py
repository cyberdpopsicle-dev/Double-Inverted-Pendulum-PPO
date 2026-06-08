import numpy as np

class DoubleInvertedPendulumEnv:
    """
    Physics Simulator for the Inverted Double Pendulum representing a Stair-Climbing Device (SCD).
    Derived from the mathematical modeling equations of Raymundo et al. (2026).
    """
    def __init__(self):
        # --- PHYSICAL PARAMETERS (Directly from Table 1 & Paper Data) ---
        self.g = 9.81         # Acceleration due to gravity (m/s^2)
        self.m1 = 0.3         # Mass of link 1 (kg)
        self.m2 = 0.3         # Mass of link 2 (kg)
        self.l1 = 0.2         # Length of link 1 (m)
        self.l2 = 0.2         # Length of link 2 (m)
        
        # Moments of inertia for uniform thin rods (J = 1/12 * m * l^2)
        self.J1 = (1.0 / 12.0) * self.m1 * (self.l1 ** 2)
        self.J2 = (1.0 / 12.0) * self.m2 * (self.l2 ** 2)
        
        # Friction/Damping parameters (Section 3.2 Viscous Damping constants)
        self.Rp1 = 1.10e-2    # Joint 1 friction coefficient (Nm*s/rad)
        self.Rp2 = 1.10e-3    # Joint 2 friction coefficient (Nm*s/rad)
        
        # --- SIMULATION CONFIGURATION ---
        self.dt = 0.002       # Timestep matching paper sampling rate (Ts = 2 ms)
        self.max_torque = 6.0 # Strict actuator constraint limit (|u| <= 6 Nm)
        
        # State vector initialization: [phi1, phi1_dot, phi2, phi2_dot]
        # Target/Goal state is OP2 (Upright balance): [pi, 0, pi, 0]
        self.state = np.array([np.pi, 0.0, np.pi, 0.0], dtype=np.float64) 

    def _get_equations_of_motion(self, state, torque):
        """
        Calculates angular accelerations (phi1_ddot, phi2_ddot) using the paper's 
        nonlinear equations of motion derived via Lagrangian Mechanics.
        """
        phi1, phi1_dot, phi2, phi2_dot = state
        M = torque
        
        # Compute passive motor friction torques
        M_R1 = self.Rp1 * phi1_dot
        M_R2 = self.Rp2 * (phi2_dot - phi1_dot)
        
        # Compute structural inertia/gravity constants (Section 3.1)
        a11 = self.m1 * ((self.l1 / 2.0) ** 2) + self.J1 + self.m2 * (self.l1 ** 2)
        a12 = self.m2 * self.l1 * (self.l2 / 2.0)
        a13 = (self.m1 * (self.l1 / 2.0) + self.m2 * self.l1) * self.g
        
        a21 = self.m2 * ((self.l2 / 2.0) ** 2) + self.J2
        a22 = self.m2 * (self.l2 / 2.0) * self.g
        
        # --- SOLVING MATRIX FORWARD DYNAMICS ---
        # Form the 2x2 Inertia Matrix H
        H11 = a11
        H12 = a12 * np.cos(phi1 - phi2)
        H21 = a12 * np.cos(phi1 - phi2)
        H22 = a21
        
        # Compute external/nonlinear force vectors
        F1 = -a12 * np.sin(phi1 - phi2) * (phi2_dot ** 2) - a13 * np.sin(phi1) + M - M_R1
        F2 =  a12 * np.sin(phi1 - phi2) * (phi1_dot ** 2) - a22 * np.sin(phi2) - M_R2
        
        # Determinant of the Inertia matrix H
        det_H = H11 * H22 - H12 * H21
        
        if np.abs(det_H) < 1e-6:
            det_H = 1e-6
            
        # Invert H matrix analytically to solve for accelerations
        phi1_ddot = (H22 * F1 - H12 * F2) / det_H
        phi2_ddot = (-H21 * F1 + H11 * F2) / det_H
        
        return phi1_ddot, phi2_ddot

    def step(self, action):
        """
        Advances the physics simulator forward by one timestep (Ts) using Runge-Kutta 4th Order.
        """
        # Enforce strict actuator limits
        torque = np.clip(float(action), -self.max_torque, self.max_torque)
        
        # Runge-Kutta 4th Order Numerical Integration for high accuracy
        k1_state = self.state
        k1_p1_dd, k1_p2_dd = self._get_equations_of_motion(k1_state, torque)
        k1_deriv = np.array([k1_state[1], k1_p1_dd, k1_state[3], k1_p2_dd], dtype=np.float64)
        
        k2_state = self.state + 0.5 * self.dt * k1_deriv
        k2_p1_dd, k2_p2_dd = self._get_equations_of_motion(k2_state, torque)
        k2_deriv = np.array([k2_state[1], k2_p1_dd, k2_state[3], k2_p2_dd], dtype=np.float64)
        
        k3_state = self.state + 0.5 * self.dt * k2_deriv
        k3_p1_dd, k3_p2_dd = self._get_equations_of_motion(k3_state, torque)
        k3_deriv = np.array([k3_state[1], k3_p1_dd, k3_state[3], k3_p2_dd], dtype=np.float64)
        
        k4_state = self.state + self.dt * k3_deriv
        k4_p1_dd, k4_p2_dd = self._get_equations_of_motion(k4_state, torque)
        k4_deriv = np.array([k4_state[1], k4_p1_dd, k4_state[3], k4_p2_dd], dtype=np.float64)
        
        # Update system state
        self.state += (self.dt / 6.0) * (k1_deriv + 2*k2_deriv + 2*k3_deriv + k4_deriv)
        
        # Continuous angle wrapping to stay within (-pi, pi] bound safely
        self.state[0] = (self.state[0] + np.pi) % (2 * np.pi) - np.pi
        self.state[2] = (self.state[2] + np.pi) % (2 * np.pi) - np.pi
        
        return np.copy(self.state)

    def reset(self, initial_state=None):
        """
        Resets environment. Defaults to unstable upright OP2 state with small noise injection.
        """
        if initial_state is not None:
            self.state = np.array(initial_state, dtype=np.float64)
        else:
            noise = np.random.uniform(-0.05, 0.05, size=4)
            self.state = np.array([np.pi, 0.0, np.pi, 0.0], dtype=np.float64) + noise
            # Force angle boundaries wraps immediately post noise injection
            self.state[0] = (self.state[0] + np.pi) % (2 * np.pi) - np.pi
            self.state[2] = (self.state[2] + np.pi) % (2 * np.pi) - np.pi
        return np.copy(self.state)