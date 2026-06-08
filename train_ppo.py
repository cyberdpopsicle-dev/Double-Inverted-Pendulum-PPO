import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from double_pendulum_gym import DoublePendulumGymEnv

def main():
    # 1. Initialize and validate the environment
    env = DoublePendulumGymEnv()
    print("Running Gymnasium structural API validation pipeline check...")
    check_env(env, warn=True)
    print("Environment verified successfully.")

    # 2. Configure hyper-parameters for robust PPO continuous optimization
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        tensorboard_log="./ppo_double_pendulum_tb/"
    )

    # 3. Learn policy behavior
    print("Commencing Deep Reinforcement Learning Agent Training (100,000 steps)...")
    model.learn(total_timesteps=100000)

    # 4. Save model weights
    model.save("ppo_double_pendulum_model")
    print("Training sequence completed successfully! Model saved as 'ppo_double_pendulum_model.zip'")

if __name__ == "__main__":
    main()