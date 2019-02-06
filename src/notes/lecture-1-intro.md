# Reinforcement Learning

## Lecture - 1: Introduction

#### TL;DR:

There are two components: _agent_ and _environment_ each having its own state

Agent receives some _observation_ and _reward_ at each step based on which it takes _action_

every RL agent has:

_policy_: a function of agent's behavior

_value function_: prediciton of expected future reward

_model_: agent's representation of environment

#### Books

- An Introduction to Reinforcement Learning, Sutton and Barto

  more brief and inutition based explanations

- Algorithms for Reinforcement Learning, Szepesvari

  more concise and mathematical in nature than intuitive

#### Features of RL

- no supervisor, only reward signal
- feedback is delayed, not instantaneous
- works on sequential, non i.i.d data
- agent's action affect the subsequent data

#### Rewards

- A reward is a **scalar** feedback signal

- it indicates how an agent is doing at a given time

- the job is to maximise cumulative reward

  **Reward Hypothesis**

  All goals can be described by maximisation of expected cumulative reward

  _What happens If there is no intermediate rewards?_

  we define an end of episode and we define a reward at the end of the episode, sum of the reward is how well he does at the end of the episode and goal is to take actions to maximise the cumulative reward at the end

  _What if the goal is time based like we want something to be done in shortest amount of time?_

  we define reward as -1 per time step and at the end of episode we maximise our goal which is minimising the cumulative negative reward

 ### Sequenetial Decision Making

- Goal: select actions to maximise total future reward
- actions may have long term consequences and reward may come later so sacrifice short term reward to gain more long-term reward

### Agent and Environment

_An example of agent and environment in atari game_

![agent-environment](./resources/agent-environment.png)

### History and State

- history is a sequence of observation, action and rewards
  $$
  H_t = A_1,O_1,R_1...A_t,O_t,R_t
  $$





- what happens next depends on history

  - agent selects action
  - environment selects observations / rewards 

- State is the summary of historic information used to determine what happens next
  $$
  S_t = f(H_t)
  $$




### Environment State

- environment state is environment's private representation

- set of information which decides what is going to happen next; environment use to pick the next observation / reward

- it is not visible to agent

- it might contain irrelevant information

  _What happens in a multi agent system?_

  An agent can consider all the other agents in environment as part of the environment. Out of scope 

### Agent State

- agent state is the agent's internal representation
- the information we capture on agent side
- agent uses this information to pick next state
- used by RL algorithms

### Information State (Markov State)

- An information state contains all useful information from history

- $$
  P[S_{t+1}|S_t] = P[S_{t+1}|S_1, S_2...S_t]
  $$

- The future is independent of the past given the present
  $$
  H_{1:t} \rightarrow S_t \rightarrow H_{t+1:\infty}
  $$

- Markov property says that you can throw away all the history if you have current state and that is sufficient to characterize the future

  _You are throwing away all previous states, How does it reconcile the fact that reward for actions may be delayed and?_

  According to Markov property it is said that the information of all previous histories is stored but we still need to look at observations and take actions based on current state and these decisions can still be optimal.

### Fully Observable Environment

- agent directly observes environment state
  - agent state = environment state = information state
  - this formalisation is Markov Decision Process

### Partially Observable Environment

- agent indirectly observes environment

- agent state != environment state

- this formalisation is Partially Observable Markov Decision Process

- agent must construct its own state representation

  - build using complete history - naive approach

  - $$
    S_t^a =H_t
    $$

  - Beliefs of environment state

  - $$
    S_t^a = (P[S_t^e=s^1],...,P[S_t^e=s^n])
    $$

  - RNN:

  - $$
    S_t^a = \sigma(S_{t-1}^aW_s + O_tW_\sigma)
    $$

    ​

### Major Components of an RL Agent

- policy: agent's behavior function; how an agent goes from it's state to what action it is going to take
- value function: how good is it to be in particular state/ take an action
- model: how the agent thinks environment works

### Policy

- agent's behavior, it is a mapping from state to action

- deterministic policy: 
  $$
  a=\gamma(s)
  $$
  we want to learn this policy and make it something which maximises the possible reward

- stochastic policy: probabilty to take a particular action conditioned on a state

  ​	
  $$
  \gamma(a|s)=P[A=a|S=s]
  $$




### Value Function

- prediction of expected future reward

- to choose between states/actions: how much total reward is it going to bring
  $$
  v_{\gamma}(s) = E_{\gamma}[R_t+\gamma^2R_{t+1}+...| S_t=s]
  $$
  value function at a state depends on the way you behave (policy) so we index it with `gamma`. We look at expectation of reward at different time steps with discounting which suggests that we care about more recent rewards than past ones.

### Model

- is not the environment itself but a model of it, it predicts what environment will do next.

- Transitions: `P` predicts what the next state is going to be i.e. dynamics of the environment

  - State Transistion Model: Probability of being in next state given current state and action

  - $$
    P_{ss^{'}}^a = Prob[S^{'}=s' | S=s,A=a]
    $$

- Rewards: `R` predicts the next _immediate_ reward

  - Reward Transistion Model: probabilty of expected reward given current state and action

    - $$
      R_{ss^{'}}^a = Prob[R | S=s,A=a]
      $$




### Categorizing RL Agents

Taxonomy -1 of RL algorithms based on which of these an agent contains

- Value Based
  - No Policy (Implicit)
  - Value Function
- Policy Based
  - Policy
  - No Value Function
- Actor Critic
  - Policy
  - Value Function

Taxonomy -2 of RL algorithms based on Model

- Model Free

  - Policy and/or Value Function
  - No Model

  We don't try to explicitly model the environment and understand how environment works

- Model Based

  - Policy and/or Value Function
  - Model

  Build the dynamics of environment and then work with it

### Problems within RL

two fundamental problems in sequential decision making

- RL
  - environment is initially unknown
  - agent interacts with environment
  - agent improves the policy
- Planning
  - A model of the environment is known
  - we tell all the dynamics of environment
  - agent performs internal computations with its model without any external interaction
  - agent improves its policy

### Exploration and Exploitation

- RL is like trial and error, agent should discover a good policy from its experiences of environment
- Exploration: finds more information about environment by giving up some reward thinking about long term
- Exploitation: exploit known information to maximise reward

### Prediction and Control

- Prediction: how well will i do given my current policy

  ​	evaluate the future given a policy

  what is the value function for uniform random policy - prediction problem

- Control: what is the optimal policy that you can maximse reward

  ​	optimise the future 

  what is the optimal value function over all possible policies? what is the optimal policy? - optimisation problem

We need to solve the prediction problem to solve control problem

