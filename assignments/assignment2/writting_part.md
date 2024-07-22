1. What are three key differences between the DQN and Q-learning algorithms?

- Use Neural Networks to approximate the Q-value function.
- Use experience replay to store and sample experiences.
- Use target networks to stabilize learning.

2. When using DQN with a deep neural network, which of the above components would you hypothesize contributes most to performance gains? Justify your answer.

The use of experience replay is the most important component of DQN. Experience replay allows the agent to learn from past experiences, which can help stabilize learning and reduce the correlation between samples. This can lead to more efficient learning and better performance.

3. In DQN, the choice of target network update frequency is important. What might happen if the target network is updated every $10^{15}$ steps for an agent learning to play a simple Atari game like Pong?

Slow target network updates can lead to instability in learning. If the target network is updated every $10^{15}$ steps, the Q-values used to calculate the target values will be outdated and inaccurate. This can lead to poor performance and slow learning.