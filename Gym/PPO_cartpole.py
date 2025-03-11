import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import os

MODEL_DIR = "models"
PPO_MODEL_DIR = f"{MODEL_DIR}/ppo_cartpole"
TRAIN = False


def get_env_info(env):
    """Print some info about the env"""
    print("Environment info")

    print("Observation space: ", env.observation_space)
    print("Example initial observation: ", env.reset()[0])
    print("Action space: ", env.action_space)


def train(env):
    """Train a model on environment, with PPO, and save"""
    # Initialize the PPO model
    print("Training model with PPO...")
    model = PPO("MlpPolicy", env, verbose=1)

    # Train the agent
    model.learn(total_timesteps=100000)

    # Save the model
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = PPO_MODEL_DIR
    model.save(model_path)
    print(f"Model saved in {model_path}.zip")

    return model


def test(model):
    """Test the trained agent"""
    print("Testing model with PyGame...")
    env = gym.make("CartPole-v1", render_mode="human")

    obs, _ = env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        env.render()
        if done or truncated:
            obs, _ = env.reset()


if __name__ == "__main__":
    # Create the environment (CartPole is a classic RL benchmark)
    env = make_vec_env("CartPole-v1", n_envs=4)  # Vectorized env for parallel training

    get_env_info(env)

    # train or load the model
    if TRAIN:
        model = train(env)
    else:
        try:
            model = PPO.load(PPO_MODEL_DIR)
        except Exception as e:
            print(f"Error loading model from {PPO_MODEL_DIR}")

    # test the model
    test(model)

    env.close()
