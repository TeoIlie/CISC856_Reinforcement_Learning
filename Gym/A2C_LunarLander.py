import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
import os

MODEL_DIR = "models"
A2C_MODEL_DIR = f"{MODEL_DIR}/a2c_lunarlander"
TRAIN = False


def get_env_info(env):
    """Print some info about the env"""
    print("Environment info")

    print("Observation space: ", env.observation_space)
    print("Example initial observation: ", env.reset()[0])
    print("Action space: ", env.action_space)


def train(env):
    """Train a model on environment, with A2C, and save"""
    # Initialize the A2C model
    print("Training model with A2C...")
    model = A2C("MlpPolicy", env, verbose=1, learning_rate=0.0007)

    # Train the agent
    model.learn(total_timesteps=100000)

    # Save the model
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = A2C_MODEL_DIR
    model.save(model_path)
    print(f"Model saved in {model_path}.zip")

    return model


def test(env, model):
    test_env = gym.make("LunarLander-v3", render_mode="human")
    obs, _ = test_env.reset()

    # Run the agent for 1000 steps or until done
    for _ in range(1000):
        action, _states = model.predict(
            obs, deterministic=True
        )  # Use deterministic policy for testing
        obs, reward, done, truncated, info = test_env.step(action)
        test_env.render()  # Display the environment
        if done or truncated:  # Reset if episode ends
            obs, _ = test_env.reset()


if __name__ == "__main__":
    # Create the environment
    env = make_vec_env("LunarLander-v3", n_envs=4)

    get_env_info(env)

    # train or load the model
    if TRAIN:
        model = train(env)
    else:
        try:
            model = A2C.load(A2C_MODEL_DIR)
        except Exception as e:
            print(f"Error loading model from {A2C_MODEL_DIR}")

    # test the model
    test(env, model)

    env.close()
