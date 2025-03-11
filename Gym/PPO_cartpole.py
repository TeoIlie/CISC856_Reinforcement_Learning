import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


def train(env):
    """Train a model on environment, with PPO"""
    # Initialize the PPO model
    model = PPO("MlpPolicy", env, verbose=1)

    # Train the agent
    model.learn(total_timesteps=100000)

    # Save the model
    model.save("ppo_cartpole")

    return model


def test(env, model):
    # Test the trained agent
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

    # train the model
    model = train(env)

    # test the model
    test(env, model)

    env.close()
