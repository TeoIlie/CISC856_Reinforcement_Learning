import gymnasium as gym
import os
import click


from stable_baselines3 import PPO
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from abc import ABC, abstractmethod

TRAIN = False

MODEL_DIR = "models"
PPO_MODEL_DIR = f"{MODEL_DIR}/ppo_cartpole"
A2C_MODEL_DIR = f"{MODEL_DIR}/a2c_lunarlander"


class GymEnvironment(ABC):
    def __init__(self, train):
        """init method must define env, model.
        train = True trains a model from scratch, else loads it from folder"""
        self.env = self.get_train_env()
        self.output_env_info()

        if train:
            self.train()
        else:
            self.load_model()

    """Abstract methods"""

    @abstractmethod
    def get_train_env(self):
        pass

    @abstractmethod
    def get_test_env(self):
        pass

    @abstractmethod
    def get_model(self):
        pass

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def get_dir(self):
        pass

    """Default methods"""

    def output_env_info(self):
        """Outputs some info about the env"""
        print("Environment info")

        print("Observation space: ", self.env.observation_space)
        print("Example initial observation: ", self.env.reset()[0])
        print("Action space: ", self.env.action_space)

    def train(self):
        """Train a model on environment and save"""
        # Initialize the model
        print("Training model...")
        self.model = self.get_model()

        # Train the agent
        self.model.learn(total_timesteps=100000)

        # Save the model
        os.makedirs(MODEL_DIR, exist_ok=True)
        model_path = self.get_dir()
        self.model.save(model_path)
        print(f"Model saved in {model_path}.zip")

    def test(self):
        """Test the trained agent"""
        print("Testing model with PyGame...")
        test_env = self.get_test_env()

        obs, _ = test_env.reset()
        for _ in range(1000):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, done, truncated, _ = test_env.step(action)
            test_env.render()
            if done or truncated:
                obs, _ = test_env.reset()


class PPOCartpole(GymEnvironment):
    def get_train_env(self):
        return make_vec_env("CartPole-v1", n_envs=4)

    def get_test_env(self):
        return gym.make("CartPole-v1", render_mode="human")

    def get_model(self):
        return PPO("MlpPolicy", self.env, verbose=1)

    def get_dir(self):
        return PPO_MODEL_DIR

    def load_model(self):
        try:
            self.model = PPO.load(PPO_MODEL_DIR)
        except Exception as e:
            print(f"Error loading model from {PPO_MODEL_DIR}")


class A2CLunarLander(GymEnvironment):
    def get_train_env(self):
        return make_vec_env("LunarLander-v3", n_envs=4)

    def get_test_env(self):
        return gym.make("LunarLander-v3", render_mode="human")

    def get_model(self):
        return A2C("MlpPolicy", self.env, verbose=1, learning_rate=0.0007)

    def get_dir(self):
        return A2C_MODEL_DIR

    def load_model(self):
        try:
            self.model = A2C.load(A2C_MODEL_DIR)
        except Exception as e:
            print(f"Error loading model from {A2C_MODEL_DIR}")


if __name__ == "__main__":
    env_options = {"l": "LunarLander", "c": "Cartpole"}
    print(f"Select one of {env_options}")
    env_type = click.prompt("Choose one", type=click.Choice(env_options.keys()))

    train_options = {"y": True, "n": False}
    print(f"Select one of {train_options}")
    train = train_options[
        click.prompt("Choose one", type=click.Choice(train_options.keys()))
    ]

    if env_type == "l":
        env = A2CLunarLander(train)
    elif env_type == "c":
        env = PPOCartpole(train)

    env.test()
