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

### Markov Process

- is a memoryless random process that we are sampling from, iteratively.


- it is sequence of states with markov property

- requires a state space and a transition probability matrix

  ​

  _how do we deal with modifications to transition probabilites over time?_

  - you can have non stationary MDPs, use same kind of algorithms that we use of stationary case but incrementally adjust your solution algorithms to find best solution you have got so far


  - augment the state to break it into different counter of states; you could have been in state once, twice, etc

### Markov Reward Processes

- Markov chain with value judgements

- (S,P,R,$\gamma$)

- S is a finite set of state

- P is transition probbaility matrix

- R is reward function; it is how much immediate reward do we get; goal is to maximise this cummulative reward (sum of this over time)
  $$
  R_s = E[R_{t+1}|S_t=s]
  $$

- $\gamma$ is a discount factor, $\gamma = [0,1]$

### Return

The return $G_t$ is the total discounted reward from time-step t into future till infinity and we make it finite my using a discount factor.
$$
G_t = R_{t+1} + \gamma R_{t+2} + ... =  \sum_{k=0}^{\infty} \gamma ^k R_{t+k+1}
$$
_Why is there no Expectation here?_

because we are talking about random sample. G is just one sample from Markov Reward Process of the reward going through one sequence.

- discount factor tells us if we like the present value of future rewards. Tells you how much I care now about the rewards I will get in future.
- If you have discount factor 0 it is maxmally short sighted that you care about only current reward. If you have discount factor of 1, that is maximally far sighted. It means that you care about all rewards going far into the future.
-  The value of receiving reward R after k+1 time steps is $\gamma ^k R$

### Why do we use discount factors?

- Mathematically conveninent
- avoid infinite returns in cyclic Markov process
- Uncertainity. We do not have the perfect model of the environment, we don't entirely trusts our evaluations going far in the future. 
- it is possible to use undiscount Markov reward in cases where a sequences terminate

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

### Markov Decision Process

