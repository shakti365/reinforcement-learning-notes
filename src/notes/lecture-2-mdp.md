#  Lecture 2: Markov Decision Process

### Introduction to MDPs

- Markov decision processes formally describes a fully observable environment for RL
- partially observable problem can be converted into MDPs

### Markov Property

$$
P[S_{t+1}|S_t] = P[S_{t+1}|S_1, S_2...S_t]
$$

The current state captures all relevant information from history

### State Transition Matrix

$$
P_{ss^{'}} = P[S_{t+1}=s^{'}|S_t=s]
$$

State Transition matrix P defines transition probabilities from state _s_ to successive states $s_{'}$
$$
P=from\begin{bmatrix}
\\P_{11}...P_{1n}
\\...
\\P_{n1}...P_{nn}\\\
\end{bmatrix}
$$
 Each row of the matrix sums to 1

## Markov Process

- is a memoryless random process that we are sampling from, iteratively.


- it is sequence of states with markov property

- requires a state space and a transition probability matrix

  _how do we deal with modifications to transition probabilites over time?_

  - you can have non stationary MDPs, use some kind of algorithms that we use for stationary case but incrementally adjust your solution algorithms to find best solution you have got so far


  - augment the state to break it into different counter of states; you could have been in state once, twice, etc

## Markov Reward Processes

- Markov chain with value judgements

- $<S,P,R,\gamma>$ 

- S is a finite set of state

- P is transition probbaility matrix

- R is reward function; it is how much immediate reward do we get; goal is to maximise this cummulative reward (sum of this over time)
  $$
  R_s = E[R_{t+1}|S_t=s]
  $$

- $\gamma$ is a discount factor, $\gamma$ = [0,1]

### Return

The return G_t is the total discounted reward from time-step t into future till infinity and we make it finite by using a discount factor.
$$
G_t = R_{t+1} + \gamma R_{t+2} + ... =  \sum_{k=0}^{\infty} \gamma ^k R_{t+k+1}
$$
_Why is there no Expectation here?_

because we are talking about random sample. G is just one sample from Markov Reward Process of the reward going through one sequence.

- discount factor tells us if we like the present value of future rewards. Tells you how much I care now about the rewards I will get in future.
- If you have discount factor 0 it is maximally short sighted that you care about only current reward. If you have discount factor of 1, that is maximally far sighted. It means that you care about all rewards going far into the future.
- The value of receiving reward R after k+1 time steps is \gamma ^k R

### Why do we use discount factors?

- Mathematically conveninent
- avoid infinite returns in cyclic Markov process
- Uncertainity. We do not have the perfect model of the environment, we don't entirely trusts our evaluations going far in the future. 
- it is possible to use undiscounted Markov reward in cases where a sequence terminates

### Value Function

- value function is long term value of being in a state
- it is expected return starting from state s

$$
v(s)=E[G_t|s_t=s]
$$

- expectation because environment is stochastic (might go in different direction in different episodes) and we want to measure expected value from all those

### Bellman Equation for MRPs

the value function can be decomposed into two parts:

- immediate reward $R_{t+1}$

- discounted value of succesor state $\gamma v(S_{t+1})$ (value that you get from that time onwards)
  $$
  v(s) = E[G_t | S_t=s]
  $$

  $$
  = E[R_{t+1} + \gamma R_{t+2} + \gamma ^2 R_{t+3} + ...| S_t=s]
  $$

  $$
  = E[R_{t+1} + \gamma ( R_{t+2} + \gamma R_{t+3} + ... )| S_t=s]
  $$

  $$
  = E[R_{t+1} + \gamma G_{t+1} | S_t=s]
  $$

  $$
  = E[R_{t+1} + \gamma v(S_{t+1}) | S_t=s]
  $$

  ... by law of iterated expectations; expectation of one variable is the same as expectation of expectation of another variable

$$
v(s) = R_s + \gamma \sum_{s^{'} \in S} P_{ss^{'}}v(s^{'})
$$

- Matrix Form
  $$
  v = R + \gamma P v
  $$











### Solving the Bellman Equation

- linear equation; can be solved directly
  $$
  v = R+\gamma P v
  $$

  $$
  (1-\gamma P)v =R
  $$

  $$
  v =(1-\gamma P)^{-1} R
  $$

- computational complexity of inversion of matrix in $O(n^3)$ for n states so it is not possible for large markov states. we will look at efficient ways to solve this problem using dynamic programming, monte-carlo evaluation and temporal difference learning

## Markov Decision Process

adding one more piece of complexity - Actions

$<S,A,P,R, \gamma>$

- S is finite set of states

- A is finite set of actions

- P is a state transition matrix
  $$
  P_{ss^{'}}=P[S_{t+1}=s^{'}|S_t=s|A_t=a]
  $$

- R is a reward function
  $$
  R_s^a=E[R_{t+1}|S_t=s,A_t=a]
  $$

- gamma is discount factor

### Policy

- a policy is distribution over actions given states such that it maximizes the reward from current state

- it depends only on the state that we are in

- policies remain same at all time-step (stationary)
  $$
  A_t \sim \pi (.|S_t; \forall t>0)
  $$





### MP and MRP can be recovered from a MDP

- Markov Decision Process $<S,P,R,\gamma,A; \pi>$: A sequence of state, transition probabilities, discounted reward and Actions taken under a policy $\pi$

- Markov Process $<S,P^\pi>$: A sequence of states with their transition probabilities under a policy $\pi$

- Markov Reward Process $<S,P^\pi,R^\pi, \gamma>$: A sequence of state, transition probabilities and a immediate  discounted Reward under a policy $\pi$
  $$
  P_{s,s'}^{\pi} = \sum_{a \in A} \pi(A|s)P_{ss'}^a
  $$

  $$
  R_{s,s'}^{\pi} = \sum_{a \in A} \pi(A|s)R_{ss'}^a
  $$


### Value Function

**State-Value Function:** $v_\pi(s)$

state value function of an MDP is expectation of future return starting from state s and following policy $\pi$
$$
v_\pi(s) = E_\pi[G_t|S_t=s]
$$
**Action-Value Function:** $q_\pi(s,a)$

action value function of an MDP is how good is it to take and action from a particular state under a policy $\pi$.
$$
q_\pi(s,a) = E_\pi[G_t|S_t=s, A_t=a]
$$

### Bellman Expectation Equation

According to bellman equation state value function can be decomposed into two parts:

- immediate reward

- discounted value of successor state
  $$
  v_\pi(s) = E_\pi[R_{t+1}+\gamma v_\pi(s_{t+1})|S_t=s]
  $$




Similary action value function can also be broken down into two  parts:
$$
q_\pi(s,a) = E_\pi[R_{t+1}+\gamma q_\pi (s_{t+1}, a_{t+1})|S_t=s]
$$
The state value function can also be interpreted by considering a single step look-ahead like this:

<INSERT PIC HERE>

Suppose you are in a state s and now you can take an action out of {a1, a2} by following a policy $\pi$. If you take action $a_1$ thus you will end up with an action-value function given by $q_\pi(s,a_1)$ and this is under a policy $\pi(a_1|s)$. You can get similar values if you take action $a_2$.

Now, when we are calculating expected state-value function at state s we can simply sum the above two action-values that you can get from state s under the infulence of policy $\pi$. This is given mathematically as follows:
$$
v_\pi(s) = \pi(a_1|s)q_\pi(s,a_1) + \pi(a_2|s)q_\pi(s,a_2) + ...
= \sum_{a \in A}\pi(a|s)q_\pi(s,a)
$$
Just like the state-value function, action-value function can also be interpreted by a one step look-ahead like this:

<INSERT PIC HERE>

When you have performed an action $a_1$ the envrionment can blow you to any of the states with a transition probability. In order to get expected value of action-value function at a state we sum over the state-value functions for all the possible state the environment ccan blow you to, in addition to the immediate reward you get taking an action from a state.
$$
q_\pi = R_s^a + \gamma \sum_{s' \in S}P_{ss'}^a v_\pi(s')
$$
TODO: Add combination of both the equations

### Optimal Value Function

- optimal state-value function $v_*(s,a)$ is the max value function over all policies.
  $$
  v_*(s) = max ( v_\pi(s))
  $$




- optimal action-value function $q_\pi(s,a)$ is the max value function over all policies.
  $$
  q_*(s,a) = max ( q_\pi(s,a))
  $$


### Optimal Policy

- It says that the optimal policy should be at least as good as all the other policy in all states
- A policy is better than other policy if state-value function of one policy is greater than or equals to another policy over all states.

$$
\pi > \pi';   v_\pi(s) \ge v_{\pi'} (s); s \in S
$$

- iall optimal policies achieve optimal state-value and action-value functions

#### Finding an optimal policy

- max over $q_*(s,a)$ 
  $$
  \pi_*(s,a) = 1; a= argmax_{a \in A}  ( q_*(s,a)) 
  $$

- optimal policy is always deterministic



### Bellman Optimality Equation

$$
v_*(s) = max ( q_*(s,a))
$$

$$
q_*(s,a) = R_s^a + \gamma \sum_{s' \in S} P_{ss'}^a v_*(s')
$$

The equations can be substituted by each other in both the cases to yield solution over one value function
$$
v_*(s) = max (  R_s^a + \gamma \sum_{s' \in S} P_{ss'}^a v_*(s') )
$$

$$
q_*(s,a) = R_s^a + \gamma \sum_{s' \in S} P_{ss'}^a max ( q_*(s',a'))
$$

- bellman optimality equation is non-linear
- has no closed form solution
- iterative solution methods like these can be applied to solve it:
  - Value Iteration
  - Policy Iteration
  - Q-Learning
  - Sarsa