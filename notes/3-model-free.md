# Model-Free Policy Evaluation

## Monte Carlo(MC) Policy Evaluation

In previous notes, we have discussed how to evaluate a policy with bellman backup, which is a model-based method, that is, we need to know the transition model $P(s' | s, a)$ and reward function $R(s, a)$.

However, in practice, we may not have access to the model, so we need to find a model-free method to evaluate a policy.

**Monte Carlo** is a model-free method to estimate the value of a policy by averaging over many episodes. 
The core idea is to use sample returns to estimate the value of each state.

With Monte Carlo, we can evaluate a policy without knowing the dynamics of the environment, and hence it do not need to assume the Markov property. 
Monte Carlo can be applied to both episodic MDPs.

### First-visit MC vs Every-visit MC

In MC policy evaluation, the typical way is as follows:
1. Initialize $N(s)=0$ and $G(s)=0$ for all $s \in S$.  
   
   $N(s)$ is the number of times state $s$ has been visited, and $G(s)$ is the sum of returns that have been observed from state $s$ with sampling.

2. Loop for each episode(a complete trajectory):
   - Generate an episode following the policy $\pi$
      
      epoisode: $i = (s_0, a_0, r_1, s_1, a_1, r_2, ..., s_{T-1}, a_{T-1}, r_T)$
    - Define the return $G_t$ as the sum of rewards from time $t$ to the end of the episode.
      
      $G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + ... + \gamma^{T-t-1} R_T$

    - For each state $s$ in the episode:
        - using the first-visit MC method:
          - if $s$ appears in the episode for the first time:
            - $N(s) \leftarrow N(s) + 1$
            - $G(s) \leftarrow G(s) + G_t$
        - using the every-visit MC method:
            - $N(s) \leftarrow N(s) + 1$
            - $G(s) \leftarrow G(s) + G_t$
        - update estimate $V^\pi(s) = G(s)/N(s)$

### Incremental MC Policy Evaluation

In the above method, we need to store the complete episode and calculate the return for each state, which is not efficient.

We can use an incremental method to update the value function after each episode, which is more efficient:
$$V^\pi(s) = V^\pi(s) \frac{N(s) - 1}{N(s)} + \frac{G_t}{N(s)} = V^\pi(s) + \frac{1}{N(s)}(G_t - V^\pi(s)) = V^\pi(s) + \alpha(G_t - V^\pi(s))$$

where $\alpha$ is the step size, which can be a constant or a function of $N(s)$.

## Metrics about how to evaluate a policy estimation

- bias: the expected error of the estimate: $E[V^\pi(s) - v^\pi(s)]$
- variance: the expected squared error of the estimate: $E[(V^\pi(s) - E[V^\pi(s)])^2]$
- mean squared error: $E[(V^\pi(s) - v^\pi(s))^2] = bias^2 + variance$

- unbiased: $E[V^\pi(s)] = v^\pi(s)$
- consistent: $V^\pi(s) \rightarrow v^\pi(s)$ as $N(s) \rightarrow \infty$

Notice that an unbiased estimator is not necessarily consistent, since consistency focuses on the convergence of the estimate to the true value as the number of samples increases.

### properties of MC policy evaluation

- first-visit MC
    - unbiased estimator
- every-visit MC
    - biased estimator
    - consistent estimator and often has better MSE
- incremental MC
    - depends on the step size $\alpha$(also known as learning rate)
      
      however, the $\alpha$ should satisfy the following conditions to ensure convergence:
        - $\sum_{t=1}^{\infty} \alpha_t(s) = \infty$
        - $\sum_{t=1}^{\infty} \alpha_t(s)^2 < \infty$

### Limitations of MC policy evaluation

- generally high variance: reduce variance requires a large number of samples, which may be impractical in some cases where data is hard to collect.
- requires episodic setting: episode must end before the value of states can be estimated, which may not be suitable for some continuous tasks.

Nevertheless, MC policy evaluation is still a powerful tool in practice, even sometime the true dynamics of the environment is known.

## Temporal Difference(TD) Learning

As combination of Monte Carlo& Dynamic programming, TD is a **model-free** method and can be applied to episodic or infinite non-episodic MDPs.

In incremental every-visit MC, we update the value function after each episode like:
$$ V^\pi(s) = V^\pi(s) + \alpha(G_t - V^\pi(s))$$

However, if we can estimate $G_t$ without waiting until the end of the episode, we can update the value function after each step, which is the core idea of TD learning.
$$ V^\pi(s) = V^\pi(s) + \alpha([r_t + \gamma V^\pi(s_{t+1})] - V^\pi(s))$$

where $r_t + \gamma V^\pi(s_{t+1})$ is the TD target, and $r_t + \gamma V^\pi(s_{t+1}) - V^\pi(s)$ is the TD error.
It is worth noting that with TD learning, we do not need episodic setting, and we can update the value function after each step, that is (s, a, r, s') tuple.

TD is a **biased** estimator, and can be **consistent** if the step size $\alpha$ satisfies the same conditions as incremental MC.
TD generally lower variance than MC, and can be more efficient in practice.


## MLE MDP Model Estimates

Even though we do not have access to the true model of the environment, we can still estimate the model from the data we have collected. 
Notice that we can do this, but some times we do not need to estimate the model, like above TD learning.

Basically, we use MLE to estimate the transition model $P(s' | s, a)$ and reward function $R(s, a)$ from the data we have collected.
Then we can use the estimated model to evaluate a policy with dynamic programming methods.

MLE MDP model estimates can have a high data efficient but very computationally expensive, and it is a consistent estimator.

## Batch MC and TD

Suppose we have the following data with 2 states A,B and $\gamma=1$, given 8 episodes:
- A,0,B,0
- B,1 (x 6)
- B,0

Then we want to estimate the value of each state with MC and TD.

In MC, $V(B) = \frac{6}{8} = 0.75$, $V(A) = \frac{0}{8} = 0$.
In TD(0), $V(B) = 0.75$, since there is no next state after B, however, $V(A) = V(A) + \alpha(0 + V(B) - V(A)) = 0 + 0.75 = 0.75$.

The core difference between MC and TD(0) in batch setting is that how they use collected data to optimize the value function.

In MC, we use the complete episode to estimate the value function, aiming to minimize the error between the observed return and the estimated value, that is, we want to minimize the MSE of the value function.

However, in TD(0), we focus more on state transitions, aiming to better predict the next state value, that is like a maximum likelihood estimation of the model, and we want to minimize the TD error.
Hence, TD(0) is same as MLE MDP model estimates.

## $\epsilon$-greedy policies

In Monte Carlo and TD learning, if the policy is deterministic, there is a big issue that the agent will not explore the environment enough, since it always chooses the maximum value action.
Without enough exploration, the monte carlo and TD learning may not converge to the optimal policy.

$\epsilon$-greedy policies are a simple way to balance exploration and exploitation. The core idea is to choose the best action with probability $1-\epsilon$ and choose a random action with probability $\epsilon$.

### GLIE: Greedy in the Limit with Infinite Exploration

GLIE is a convergence condition for $\epsilon$-greedy policies, which ensures that the agent will explore the environment infinitely often, and the policy will converge to the optimal policy.
GLIE requires that:
1. all state-action pairs are visited infinitely: $\lim_{k \rightarrow \infty} N_k(s, a) = \infty$
2. the policy converges to a greedy policy: $\lim_{k \rightarrow \infty} \pi_k(a|s) \rightarrow \delta(a = \arg\max_a Q_k(s, a))$ with probability 1

In practice, we can use a decaying $\epsilon$ to satisfy the GLIE condition, like $\epsilon_k = \frac{1}{k}$.

It is a theoretical guarantee that **GLIE** Monte Carlo control converges to the optimal policy.

### SARSA and Q-learning

SARSA and Q-learning are also TD learning methods, but they are more focused on learning the action-value function $Q(s, a)$.
While above TD(0) is more focused on learning the state-value function $V(s)$.

SARSA is an on-policy TD control method, which updates the action-value function $Q(s, a)$ with the following rule:
$$Q(s, a) = Q(s, a) + \alpha(r + \gamma Q(s', a') - Q(s, a))$$
where $a'$ is the action chosen by the policy $\pi$ in state $s'$.

SARSA for finite-state and finite-action MDPs converges to the optimal action-value under the following conditions:
1. Policy sequence $\pi_t(a|s)$ satisfies the GLIE condition
2. The step size $\alpha_t(s, a)$ satisfies the Robbins-Monro conditions:
    - $\sum_{t=1}^{\infty} \alpha_t(s, a) = \infty$
    - $\sum_{t=1}^{\infty} \alpha_t(s, a)^2 < \infty$
For example, we can use $\alpha_t = \frac{1}{T(s, a)}$.


Q-learning is an off-policy TD control method, which updates the action-value function $Q(s, a)$ with the following rule:
$$ Q(s, a) = Q(s, a) + \alpha(r + \gamma \max_{a'} Q(s', a') - Q(s, a))$$
where $a'$ is the action that maximizes the action-value function $Q(s', a')$.

Notice that:
on-policy means mechanism learn to estimate and evaluate a policy from experience obtained from following that policy.
while off-policy means mechanism learn to estimate and evaluate a policy from experience obtained from following a different policy.

## Function Approximation

In practice, the state space of the environment may be very large or continuous, so it is impossible to store the value function for each state.

Function approximation enables us to approximate the value function with a parameterized function, like a neural network, which can avoid explicitly storing the value function for each state and hence have a more compact representation and higher calculation and learning efficiency.
Moreover, function approximation can generalize the value function to unseen states, which is very important in practice.

The core idea of function approximation is to learn the parameters of the function approximator to minimize the error between the predicted value and the observed return, like a supervised learning problem.
However since we do not have a 'oracle' to provide the true value, we need to use the TD or MC error as the target to train the function approximator.

However there is a *deadly triad* in VFA, that is:
- Bootstrapping: using the estimated value to update the value function
- Function approximation: using a parameterized function to approximate the value function
- Off-policy learning: learning the value function from experience obtained from following a different policy

When theses three are combined, the convergence of the value function is not guaranteed, and the value function may diverge. For example, VFA may lead to inaccuracy, which will be amplified by bootstrapping, and may be further amplified by off-policy learning due to the mismatch between the target policy and the behavior policy.

## DQN

Typically, Q-learning can converge to the optimal action-value function with a tabular representation, but it may not converge with VFA since all state-action value functions are sharing the same parameters.
Hence 2 issues arise:
1. Correlation between samples: these samples are not independent and identically distributed, which may lead to divergence when using them to update the single shared value function.
2. Non-stationary target: the target value is changing as the value function is updated, which may lead to divergence.

And DQN is a solution to these issues, which uses the following techniques:
- Experience replay
- Fixed Q-targets

### Experience replay

Experience replay is a technique to store the agent's experience in a replay buffer, and sample a mini-batch of experiences from the replay buffer to update the value function.
The core idea is to break the correlation between samples and make the samples more independent and identically distributed. And it can also improve the data efficiency and learning stability.

### Fixed Q-targets

Fixed Q-targets is a technique to use a separate target network to calculate the target value, and update the target network parameters less frequently than the value network parameters.