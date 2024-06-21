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