import matplotlib.pyplot as plt
import numpy as np

episode_rewards = []

isCumulative = True
filename = 'figures/reward_300_embed2.txt'
with open(filename, 'r') as file:
    for line in file:
        episode_rewards.append(float(line.strip()))

if isCumulative:
    cumulative_rewards = np.cumsum(episode_rewards)
    plt.plot(cumulative_rewards)
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.title('Episode Cumulative Rewards')
    plt.grid(True)

else:
    plt.plot(episode_rewards)
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.title('Episode Rewards')
    plt.grid(True)

# Save plot as an image file
plt.savefig('figures/episode_300_embed2_simple.png')
plt.show()