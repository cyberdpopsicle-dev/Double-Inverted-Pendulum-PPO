import gymnasium as gym
from gymnasium import spaces
import numpy as np
from double_pendulum_env import DoubleInvertedPendulumEnv

class DoublePendulumGymEnv(gym.Env):
    """
    Custom Farama Gymnasium Wrapper for the Inverted Double Pendulum Simulator.
    Enables plug-and-play compatibility with RL libraries like Stable-Baselines3.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super(DoublePendulumGymEnv, self).__init__()
        
        self.underlying_env = DoubleInvertedPendulumEnv()
        
        # Continuous single motor control torque input: u ∈ [-6.0, 6.0] Nm
        self.action_space = spaces.Box(
            low=-float(self.underlying_env.max_torque),
            high=float(self.underlying_env.max_torque),
            shape=(1,),
            dtype=np.float32
        )
        
        # State vector metrics bounds configuration: [phi1, phi1_dot, phi2, phi2_dot]
        high_obs = np.array([
            np.pi,   # phi1 max
            25.0,    # phi1_dot max
            np.pi,   # phi2 max
            25.0     # phi2_dot max
        ], dtype=np.float32)
        
        self.observation_space = spaces.Box(
            low=-high_obs,
            high=high_obs,
            dtype=np.float32
        )
        
        self.current_step = 0
        self.max_episode_steps = 500  # 500 steps * 2ms = 1.0 second per episode

    def reset(self, seed=None, options=None):
        """
        Resets the physical state environment to initiate a new training episode.
        """
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
            
        raw_state = self.underlying_env.reset()
        self.current_step = 0
        
        return raw_state.astype(np.float32), {}

    def step(self, action):
        """
        Processes an agent step execution command. Updates physics and evaluates reward.
        """
        self.current_step += 1
        
        # Cleanly extract action scalar to prevent nesting errors
        if isinstance(action, (list, np.ndarray)):
            torque_input = float(action[0])
        else:
            torque_input = float(action)
            
        # Advance physics engine
        next_state = self.underlying_env.step(torque_input)
        phi1, phi1_dot, phi2, phi2_dot = next_state
        
        # --- REWARD FUNCTION DESIGN ---
        # Goal: Target upright convergence configurations at OP2: phi1=pi, phi2=pi
        # Compute shortest angular errors relative to targeted upright target (pi)
        error_phi1 = np.arctan2(np.sin(phi1 - np.pi), np.cos(phi1 - np.pi))
        error_phi2 = np.arctan2(np.sin(phi2 - np.pi), np.cos(phi2 - np.pi))
        
        # Quadratic tracking deviations penalties
        reward_tracking = -(8.0 * (error_phi1 ** 2) + 8.0 * (error_phi2 ** 2))
        # Velocity stabilization dampening penalties
        reward_velocity = -(0.1 * (phi1_dot ** 2) + 0.1 * (phi2_dot ** 2))
        # Control actuation effort penalization
        reward_control = -0.01 * (torque_input ** 2)
        
        reward = reward_tracking + reward_velocity + reward_control
        
        # --- TERMINATION & TRUNCATION DETECTION ---
        truncated = self.current_step >= self.max_episode_steps
        terminated = False
        
        # Crash termination threshold if links fall past salvageable bounds
        if abs(error_phi1) > (np.pi / 2.0) or abs(error_phi2) > (np.pi / 2.0):
            terminated = True
            reward -= 150.0  # Big crash penalty punishment
            
        # Clip observation variables safely to fit exactly into observation space boundaries
        clipped_obs = np.clip(next_state, self.observation_space.low, self.observation_space.high)
        
        return clipped_obs.astype(np.float32), float(reward), terminated, truncated, {}