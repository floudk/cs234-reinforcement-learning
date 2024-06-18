# Assignment 1

## 1. Effect of Effective Horizon

Consider an agent managing inventory for a store, which is represented as an MDP.  
stock level s: [0, 10] refers to the number of items currently in stock.  
action a: sell(1 item if stock > 0) or buy(1 item if stock < 10).
rewards r:
- valid sell: +1
- reach s = 10: +100
terminal state: s = 10
start state: s = 3

We will consider how the agent’s optimal policy changes as we adjust the finite horizon H of the problem.

### a. Starting from the initial state s = 3, is it possible to choose a value of H that results in the optimal policy taking both buy and sell steps during its execution? Explain why or why not.

It is possible when 3 < H < 7, in this case, the agent can not reach the terminal state s = 10, and the optimal policy will first try to sell all the items in stock in first 3 steps, then buy and sell alternatively to get the maximum reward.


### b. For what values of H does the optimal policy reach a fully stocked inventory, starting from the initial state s = 3? I.e. Give a range for H. 
Note 1: we consider the inventory fully stocked if a buy action is chosen in state s = 9, causing a transition to s = 10. This includes the last time step in the horizon.   
Note 2: By executing only buy actions, the agent can reach s = 10 from s = 3 in H = 7 steps.  

H >= 7, the optimal policy will reach a fully stocked inventory.


### c. Now consider the infinite-horizon discounted setting. That is, there is no time limit – the problem can only terminate when a terminal state is reached. Suppose γ = 0. What action does the optimal policy take when s = 3? What action does the optimal policy take when s = 9?

When γ = 0, the agent only cares about immediate rewards, so when s = 3, the optimal policy will take the sell action, and when s = 9, the optimal policy will take the buy action to reach the terminal state s = 10.

### d. In the infinite-horizon discounted setting, is it possible to choose a fixed value of γ ∈ [0, 1) such that the optimal policy starting from s = 3 never fully stocks the inventory? You do not need to propose a specific value, but simply explain your reasoning either way.

yes, like in the case of γ = 0, the agent only cares about immediate rewards, no matter when the agent in state 1~8, it will directly sell the item to get the reward, so it will never fully stock the inventory.


## 2. Reward Hacking

A typical reinforcement learning algorithm in autonomous driving is trying to minimize the mean commute time of all drivers, which can be simplified as maximizing the mean velocity of all vehicles.   
However, under this reward, thje optimal policy of a single AI car is to park and not merge onto the highway, which is not the desired behavior.

### a. Explain why the optimal policy for the AI car is not to merge onto the highway

If the car merges onto the highway, it will slow down the whole traffic flow behind it, which will decrease the mean velocity of almost all vehicles, so the optimal policy for the AI car is not to merge onto the highway.
In a word, the increase of single car's velocity will not compensate for the decrease of the mean velocity of all vehicles.

### b. Note this behavior is not aligned with the true reward function. Share some ideas about alternate reward functions (that are not minimizing commute) that might still be easier to optimize, but would not result in the AI car never merging. 
Your answer should be 2-5 sentences and can include equations: there is not a single answer and reasonable solutions will be given full credit.

May be an average value between mean velocity and the lowest velocity of all vehicles.
Like $R = \alpha \bar{v} + (1 - \alpha) \min(v)$, by adjusting the value of $\alpha$, we can balance the mean velocity and the lowest velocity of all vehicles, and the AI car will not just park and not merge onto the highway.


## 3. Bellman Residuals and performance bounds

Bellman Backup Operator is defined as follows:
$$ BV(s) = max_a \{ r(s,a) + \gamma \sum_{s'\in S} p(s'|s,a) V(s')\} $$

And the contraction operator $B^\pi$ with the fixed point $V^\pi$, which is the Bellman backup operator for a particular policy $\pi$:
$$ B^\pi V(s) = r(s, \pi(s)) + \gamma \sum_{s'\in S} p(s'|s,\pi(s)) V(s') $$
Notice that in the above equation, we do not take the maximum over actions, as we are evaluating a concrete policy.

In this case, we assume $\pi$ is deterministic, and we have showed in class that $$||BV - BV' || \leq \gamma ||V - V'||$$ for any two value functions $V$ and $V'$.



### a. show that the analogous inequality $||B^\pi V - B^\pi V' || \leq \gamma ||V - V'||$ also holds.

$$ || B^\pi V - B^\pi V' || = || r(s, \pi(s)) + \gamma \sum_{s'\in S} p(s'|s,\pi(s)) V(s') - r(s, \pi(s)) - \gamma \sum_{s'\in S} p(s'|s,\pi(s)) V'(s') || $$
$$ = \gamma || \sum_{s'\in S} p(s'|s,\pi(s)) (V(s') - V'(s')) || \leq \gamma \sum_{s'\in S} p(s'|s,\pi(s)) || V(s') - V'(s') || $$
since given $S$, $p(s'|s,\pi(s))$ is a probability distribution, so we have $\sum_{s'\in S} p(s'|s,\pi(s)) = 1$, so we have:
$$ \leq \gamma || V - V' || $$

### b. Prove that the fixed point for $B^\pi$ is unique.

Suppose there are two fixed points $V_1$ and $V_2$ for $B^\pi$, then we have:
$V_1 = B^\pi V_1$, $V_2 = B^\pi V_2$.
Then we have:
$$ || V_1 - V_2 || = || B^\pi V_1 - B^\pi V_2 || \leq \gamma || V_1 - V_2 || $$
which means $|| V_1 - V_2 || = 0$, so the fixed point for $B^\pi$ is unique.

### c. Suppose that V and V' are vectors satisfying $V(s) \leq V'(s)$ for all states s. Show that $B^\pi V(s) \leq B^\pi V'(s)$ for all states s.

To show $B^\pi V(s) \leq B^\pi V'(s)$, we need to show $r(s, \pi(s)) + \gamma \sum_{s'\in S} p(s'|s,\pi(s)) V(s') \leq r(s, \pi(s)) + \gamma \sum_{s'\in S} p(s'|s,\pi(s)) V'(s')$.
which can be further simplified as:
$$ \sum_{s'\in S} p(s'|s,\pi(s)) V(s') \leq \sum_{s'\in S} p(s'|s,\pi(s)) V'(s') $$

And given that $V(s) \leq V'(s)$ for all states s, we have:
$$ \sum_{s'\in S} p(s'|s,\pi(s)) V(s') \leq \sum_{s'\in S} p(s'|s,\pi(s)) V'(s') $$

## 3.1 Bellman Residuals

After gaining some intuition for value functions and Bellman operators, we will now explore **how policies can be extracted and what their performance might look like**.

A straightforward way to extract a greedy policy $\pi$ from an arbitrary value function $V$ can use:
$$ \pi(s) = \arg\max_a \{ r(s,a) + \gamma \sum_{s'\in S} p(s'|s,a) V(s')\} $$
To better understand the performance of the greedy policy, we can define the following quantity:
- **Bellman Residuals**: $BV(s) - V(s)$, which can be used to evaluate whether the value function $V$ is close to the optimal value function $BV$, since the optimal value function is a fixed point of the Bellman operator, that is $BV = V$, so the residuals should be close to zero.
- **Bellman error magnitude**: $||BV - V||\infty$, which is the maximum absolute value of the residuals over all states, also can be used to evaluate the performance of the value function.

### d. For what value function V, is the Bellman error magnitude zero?

The Bellman error magnitude is zero when the value function V is the optimal value function $BV$, that is $V = BV$.
Hence, the Bellman error magnitude is zero when the value function V is the optimal value function.

### e. Prove the following statemenets for an arbitrary value function V and any policy $\pi$:

$$ ||V -  V^\pi || \leq \frac{||V - B^\pi V||}{1 - \gamma} $$
$$ ||V -  V* || \leq \frac{||V - BV||}{1 - \gamma} $$

The first inequality can be proved as follows:
Since $V^\pi$ is a fixed point in policy $\pi$, we have $V^\pi = B^\pi V^\pi$, then we have:
$$ ||V -  V^\pi || = ||V - B^\pi V^\pi|| \leq ||V - B^\pi V|| + ||B^\pi V - B^\pi V^\pi|| $$
Based on the contraction property of Bellman operator, we have $||B^\pi V - B^\pi V^\pi|| \leq \gamma ||V - V^\pi||$, then we have:
$$ ||V -  V^\pi || \leq ||V - B^\pi V|| + \gamma ||V - V^\pi|| $$
Then we have:
$$ ||V -  V^\pi || \leq \frac{||V - B^\pi V||}{1 - \gamma} $$

For the second inequality:
since $V*$ is the optimal value function, it also a fixed point in the optimal policy, so the proof is similar to the first inequality.

## 3.2 bound on the policy performance

### f. V is an arbitrary value function, and $\pi$ is the greedy policy with respect to V. Let $\epsilon = ||BV - V||$, which is the Bellman error magnitude for V. Prove that for any state s, the following inequality holds:
$$ V^\pi(s) \geq V*(s) - \frac{2\epsilon}{1 - \gamma} $$

since $\pi$ is the greedy policy with respect to V, we have $B^\pi V = BV$, then we can transform the (e) inequality to:

$$ ||V -  V^\pi || \leq \frac{\epsilon}{1 - \gamma} $$

$$ ||V -  V* || \leq \frac{\epsilon}{1 - \gamma} $$

Then we have:

$$ ||V^\pi - V* || \leq ||V^\pi - V|| + ||V - V*|| \leq \frac{2\epsilon}{1 - \gamma} $$

That's to say, for any state s, the following inequality holds:
$$ V^\pi(s) \geq V*(s) - \frac{2\epsilon}{1 - \gamma} $$

This inequality shows that the performance of the greedy policy from any value function V has a lower bound, which is related to the Bellman error magnitude $\epsilon$ and the discount factor $\gamma$.

### g. Give an example real-world application or domain where having a lower bound on $V^\pi(s)$ is useful.

In autonomous driving, the agent needs to make decisions based on the current state, and the lower bound on $V^\pi(s)$ can offer a quantitative measure of the performance of the greedy policy, which is a guarantee of the performance like safety, efficiency, etc. 

### h. For another value function V' and its greedy policy $\pi'$. $ || BV' - V' || = \epsilon = || BV - V || $, Does the lower bound imply that $ V^\pi(s) = V^{\pi'}(s) $ for all states s?

No, same $\epsilon$ only means the Bellman error magnitude is the same, which is kind of the optimizality of the value function, but it does not mean the value function is the same, nor the greedy policy is the same.

## 3.3 what if our algorithm returns a V that satisfies $V* \leq V$

### i. if $V* \leq V$, show the following holds for any state s:
$$ V^\pi(s) \geq V*(s) - \frac{\epsilon}{1 - \gamma} $$

Since $V* \leq V$, we have $BV* \leq BV$, then we have:
$$ V + \epsilon \geq BV \geq BV* $$ 
that is:
$$ V \geq BV* - \epsilon $$

Since $\pi$ is the greedy policy with respect to V, we have $B^\pi V = V^\pi$, then we have:
$$ V^\pi(s) \geq V*(s) - \epsilon$$

Further, since we iteratively update the value function V, and in each iteration, the Bellman error magnitude is reduced by a factor of $\gamma$, that is
$$ \epsilon + \gamma \epsilon + \gamma^2 \epsilon + \cdots = \frac{\epsilon}{1 - \gamma} $$

Then we have:
$$ V^\pi(s) \geq V*(s) - \frac{\epsilon}{1 - \gamma} $$

## 3.4 challenges

### j. $ V* \leq V$ is not easy to show since the optimal value function is unknown. However, we can show that if $BV \leq V$, then $V* \leq V$, which is much easier to show. Prove this statement.

Based on deduction:
given that $BV \leq V$, then we suppose for some number $n \geq 1$, we have $B^n V \leq V$, then we try to prove $B^{n+1} V \leq V$.

$$ B^{n+1} V = B(B^n V) \leq B V \leq V $$

and when $\lim_{n \to \infty} B^n V = V*$, we have $V* \leq V$.





### k. it is still possible to make the bounds tighter. Still, let V be an arbitrary value function, and $\pi$ is the greedy policy with respect to V, $\epsilon = ||BV - V||$. Prove that for any state s, the following inequality holds:
$$ V^\pi(s) \geq V*(s) - \frac{2\epsilon \gamma}{1 - \gamma} $$
and further if $V* \leq V$, then we have:
$$ V^\pi(s) \geq V*(s) - \frac{\epsilon \gamma}{1 - \gamma} $$

For the first inequality:
Since $V^\pi(s) = BV(s)$, then 
$$ V^\pi(s) \leq V^*(s) - \gamma \epsilon$$
when we consider the iteration, we can get
$V^\pi(s) \leq V*(s) - \frac{\epsilon \gamma}{1 - \gamma}$
however, the error should be consider twice in iteration, hence we can prove the first.

The second one is the same as above.
