import gymnasium as gym
import torch
from dqn import DQN
import numpy as np
import struct

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Initialize the environment
env = gym.make("LunarLander-v3", render_mode="human")

n_observations = env.observation_space.shape[0]
n_actions = env.action_space.n

# Initialize the model architecture
policy_net = DQN(n_observations, n_actions).to(device)

# Load the trained weights
policy_net.load_state_dict(torch.load("models/dqn_lunar_lander.pth", map_location=torch.device('cpu')))


def write_tensor(f, tensor):
    # Ensure CPU, float32, contiguous
    arr = tensor.detach().cpu().contiguous().numpy().astype(np.float32)
    count = arr.size

    print(arr)

    # write element count (int32)
    f.write(struct.pack("<i", count))

    # write raw float32 data
    arr.tofile(f)


with open("models/dqn_lunar_lander.bin", "wb") as f:
    write_tensor(f, policy_net.layer1.weight)
    write_tensor(f, policy_net.layer1.bias)

    write_tensor(f, policy_net.layer2.weight)
    write_tensor(f, policy_net.layer2.bias)

    write_tensor(f, policy_net.layer3.weight)
    write_tensor(f, policy_net.layer3.bias)
