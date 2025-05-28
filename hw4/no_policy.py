import gymnasium as gym
import time

env = gym.make("CartPole-v1", render_mode="human")
print(f"Action Space: {env.action_space}")
print(f"Observation Space: {env.observation_space}")

num_episodes = 60

for episode in range(num_episodes):
    observation, info = env.reset(seed=0)

    total_reward = 0
    terminated = False
    truncated = False

    print(f"\nEpisode {episode + 1}")

    while not terminated and not truncated:
        env.render()

        # This is the simplest possible policy - no learning involved yet!
        action = env.action_space.sample()

        observation, reward, terminated, truncated, info = env.step(action)

        # Accumulate the reward
        total_reward += reward

    print(f"Episode {episode + 1} finished.")
    print(f"Total Reward: {total_reward}")

# Close the environment when done
env.close()

print("\n--- Simulation Finished ---")
