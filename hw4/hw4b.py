import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import time
import pickle
import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.optim as optim
from scipy.signal import savgol_filter

# %% [markdown]
# ### Setup

# %%
# PPO Hyperparameters
learning_rate_actor = 0.0003
learning_rate_critic = 0.001
gamma = 0.99  # Discount factor
eps_clip = 0.2  # PPO clip parameter
epochs = 10  # Number of epochs for PPO update
batch_size = 64  # Size of a mini-batch for PPO update
update_timestep = 2000  # Update policy every n timesteps
max_ep_len = 500  # Max timesteps in one episode
episodes = 2000  # Number of training episodes
show_every = 100  # How often to print progress
render = False

# %%
# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
# Environment initialization
env = gym.make("CartPole-v1", render_mode="human" if render else None)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
print(f"Action Space: {env.action_space}")
print(f"Observation Space: {env.observation_space}")
print(f"State Dimensions: {state_dim}")
print(f"Action Dimensions: {action_dim}")

# %% [markdown]
# ### Memory Buffer


# %%
class RolloutBuffer:
    """Stores transitions collected from the environment."""

    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.state_values = []  # Store critic values

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.state_values[:]


# %% [markdown]
# ### Actor-Critic Network


# %%
class ActorCritic(nn.Module):
    """Defines the Actor-Critic network architecture."""

    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()

        # Actor Network
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1),
        )

        # Critic Network
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        """
        Selects an action based on the current state and returns action,
        log probability, and state value.
        """
        action_probs = self.actor(state)
        dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)

        return action.detach(), action_logprob.detach(), state_val.detach()

    def evaluate(self, state, action):
        """
        Evaluates the given state and action, returning log probability,
        state value, and distribution entropy.
        """
        action_probs = self.actor(state)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy


# %% [markdown]
# ### PPO Agent


# %%
class PPO:
    """Proximal Policy Optimization Agent."""

    def __init__(
        self,
        state_dim,
        action_dim,
        lr_actor,
        lr_critic,
        gamma,
        epochs,
        eps_clip,
    ):
        self.gamma = gamma
        self.epochs = epochs
        self.eps_clip = eps_clip

        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.policy.actor.parameters(), "lr": lr_actor},
                {"params": self.policy.critic.parameters(), "lr": lr_critic},
            ]
        )

        self.policy_old = ActorCritic(state_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def select_action(self, state):
        """Selects an action using the old policy for data collection."""
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            action, action_logprob, state_val = self.policy_old.act(state)

        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)

        return action.item()

    def update(self):
        """Updates the policy using PPO."""
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal, value in zip(
            reversed(self.buffer.rewards),
            reversed(self.buffer.is_terminals),
            reversed(self.buffer.state_values),
        ):
            if is_terminal:
                discounted_reward = 0
            # Use critic value as baseline if not terminal, else use 0
            # GAE (Generalized Advantage Estimation) could also be used here.
            # Here we use a simpler approach: Returns = Rewards + gamma * V(s_next)
            # If terminal, V(s_next) = 0. If not, V(s_next) is the stored critic value.
            # Since we iterate backwards, V(s_next) is `discounted_reward`.
            # We want Returns = Rewards + gamma * V(s_next)
            # So, R_t = r_t + gamma * R_{t+1} if not terminal, or r_t if terminal.
            # Alternatively, we calculate advantages: A_t = r_t + gamma * V(s_{t+1}) - V(s_t)
            # And use Returns = A_t + V(s_t)
            # Here we use Q_t ~ r_t + gamma * V(s_{t+1})
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # Convert list to tensor
        old_states = (
            torch.squeeze(torch.stack(self.buffer.states, dim=0))
            .detach()
            .to(device)
        )
        old_actions = (
            torch.squeeze(torch.stack(self.buffer.actions, dim=0))
            .detach()
            .to(device)
        )
        old_logprobs = (
            torch.squeeze(torch.stack(self.buffer.logprobs, dim=0))
            .detach()
            .to(device)
        )
        old_state_values = (
            torch.squeeze(torch.stack(self.buffer.state_values, dim=0))
            .detach()
            .to(device)
        )

        # Calculate advantages: A_t = Rewards_t - V_old(s_t)
        advantages = rewards.detach() - old_state_values.detach()

        # Optimize policy for K epochs
        for _ in range(self.epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(
                old_states, old_actions
            )

            # Match state_values tensor dimensions with rewards tensor for loss calculation
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            surr1 = ratios * advantages
            surr2 = (
                torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip)
                * advantages
            )

            # Final loss of policy + Value loss + Entropy loss
            # PPO-Clip loss + Value Function Loss - Entropy Bonus
            loss = (
                -torch.min(surr1, surr2)
                + 0.5 * self.MseLoss(state_values, rewards)
                - 0.01 * dist_entropy
            )

            # Take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # Clear buffer
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(
            torch.load(
                checkpoint_path, map_location=lambda storage, loc: storage
            )
        )
        self.policy.load_state_dict(
            torch.load(
                checkpoint_path, map_location=lambda storage, loc: storage
            )
        )


# %% [markdown]
# ### Training Loop

# %%
ppo_agent = PPO(
    state_dim,
    action_dim,
    learning_rate_actor,
    learning_rate_critic,
    gamma,
    epochs,
    eps_clip,
)
total_rewards = []
highest_reward = 0
time_step = 0
episode_count = 0

# Training loop
for episode in range(1, episodes + 1):
    state, info = env.reset()
    current_ep_reward = 0
    terminated = False
    truncated = False

    render_this_episode = episode % show_every == 0

    while not terminated and not truncated:
        # Select action with policy
        action = ppo_agent.select_action(state)
        state, reward, terminated, truncated, info = env.step(action)

        # Saving reward and is_terminals
        ppo_agent.buffer.rewards.append(reward)
        ppo_agent.buffer.is_terminals.append(
            terminated
        )  # Store terminated, not truncated

        time_step += 1
        current_ep_reward += reward

        # Render
        if render_this_episode and render:
            env.render()

        # Update PPO agent
        if time_step % update_timestep == 0:
            # Need to add the value of the last state if not terminal
            if not terminated and not truncated:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).to(device)
                    _, _, last_value = ppo_agent.policy_old.act(state_tensor)
                    ppo_agent.buffer.state_values.append(last_value)
                    ppo_agent.buffer.rewards[-1] += (
                        gamma * last_value.item()
                    )  # Add bootstrapped value
                    ppo_agent.buffer.is_terminals.append(
                        False
                    )  # Add a dummy terminal flag
            ppo_agent.update()
            time_step = 0  # Reset timestep counter after update

    total_rewards.append(current_ep_reward)
    if current_ep_reward > highest_reward:
        highest_reward = current_ep_reward

    # Logging
    if episode % show_every == 0:
        avg_reward = sum(total_rewards[-show_every:]) / len(
            total_rewards[-show_every:]
        )
        print(
            f"Episode: {episode:5} | Avg Reward (last {show_every}): {avg_reward:6.2f} | Highest: {highest_reward:5.0f}"
        )
        highest_reward = 0  # Reset highest for the next block

# %%
# Save trained model
data = {
    "policy_state_dict": ppo_agent.policy.state_dict(),
    "total_rewards": total_rewards,
}
with open("data_PPO.pkl", "wb") as f:
    pickle.dump(data, f)

# %% [markdown]
# ### Postprocessing

# %%
# Load trained model
with open("data_PPO.pkl", "rb") as f:
    data = pickle.load(f)
    policy_state_dict = data["policy_state_dict"]
    total_rewards = data["total_rewards"]

# %%
# Plot
plt.plot(savgol_filter(total_rewards, 101, 3))  # Smoothed plot
plt.plot(total_rewards, alpha=0.3)  # Raw plot
plt.title("Total Rewards per Episode (Savgol Filtered)")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.ylim([0, 600])  # CartPole-v1 max is 500, but allow some overshoot
plt.show()

# %%
# Render a single episode using the trained PPO agent
env = gym.make("CartPole-v1", render_mode="human")
ppo_agent = PPO(
    state_dim,
    action_dim,
    learning_rate_actor,
    learning_rate_critic,
    gamma,
    epochs,
    eps_clip,
)
ppo_agent.policy_old.load_state_dict(policy_state_dict)

observation, info = env.reset()
terminated = False
truncated = False

while not terminated and not truncated:
    with torch.no_grad():
        state_tensor = torch.FloatTensor(observation).to(device)
        action_probs = ppo_agent.policy_old.actor(state_tensor)
        # Select the most likely action (exploitation)
        action = torch.argmax(action_probs).item()

    next_observation, reward, terminated, truncated, info = env.step(action)
    observation = next_observation
    time.sleep(0.02)  # 50 FPS max

env.close()
