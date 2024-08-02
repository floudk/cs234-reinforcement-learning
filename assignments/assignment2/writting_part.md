### Q1

1. What are three key differences between the DQN and Q-learning algorithms?

- Use Neural Networks to approximate the Q-value function.
- Use experience replay to store and sample experiences.
- Use target networks to stabilize learning.

2. When using DQN with a deep neural network, which of the above components would you hypothesize contributes most to performance gains? Justify your answer.

The use of experience replay is the most important component of DQN. Experience replay allows the agent to learn from past experiences, which can help stabilize learning and reduce the correlation between samples. This can lead to more efficient learning and better performance.

3. In DQN, the choice of target network update frequency is important. What might happen if the target network is updated every $10^{15}$ steps for an agent learning to play a simple Atari game like Pong?

Slow target network updates can lead to instability in learning. If the target network is updated every $10^{15}$ steps, the Q-values used to calculate the target values will be outdated and inaccurate. This can lead to poor performance and slow learning.

### Q2

1. To compute the REINFORCE estimator, you will need to calculate the values {$G_t$} where $G_t = \sum_{t'=t}^{T} \gamma^{t'-t} R_{t'}$. Naively calculating all these values for each time step would task $O(T^2)$ time. Describe a more efficient way to calculate these values that requires only $O(T)$ time.

Reverse accumulation can be used to calculate the values {$G_t$} in $O(T)$ time. 


2. Consider the cases in the gradient of the clipped PPO loss function equals 0. express these cases mathematically and explain why PPO behaves in this manner.

when the gradient of the clipped PPO loss function equals 0, it means that the policy has not changed significantly. 
This can happen when the ratio is needed to be clipped or when advantage is near 0. 
That is :
$ L^{CLIP} = \mathbb{E}[\min(r_t(\theta)A_t, clip(r_t(\theta), 1-\epsilon, 1+\epsilon)A_t)]$

when the ratio is clipped, the function will be $clip(r_t(\theta), 1-\epsilon, 1+\epsilon)A_t$, and since $1-\epsilon$ and $1+\epsilon$ are constants, the gradient will be 0.

And for advantage near 0, that means in current policy, the action is at least as good as the average action, so the policy will not change.


3. Notice that the method which samples actions from the policy also returns the log-probability with
which the sampled action was taken. Why does REINFORCE not need to cache this information while
PPO does? Suppose this log-probability information had not been collected during the rollout. How
would that affect the implementation (that is, change the code you would write) of the PPO update?