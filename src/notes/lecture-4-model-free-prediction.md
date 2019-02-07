# Lecture 4: Model Free Prediction

**Model free prediction:**

- Estimate the value function of an unknown MDP when policy is given



### Monte Carlo Reinforcement Learning

- Learns from complete episodes

- Knowledge of transition probabilities or reward is NOT required

- Only works for episodic MDPs; all episodes must terminate

- **Goal:** learn $v_{\pi}$ (value function) from episodes of experience under policy $\pi$
  $$
  S_1, A_1, R_1...S_k,A_k,R_k 
  $$

- Earlier (In DP) we saw that value function was defined by expected return

  
  $$
  V_\pi = E_\pi[G_t|S_t=s]
  $$
  Now, In Monte-Carlo we relace this expectation by empirical mean.



> **empirical (arithmetic) mean** is the total of samples divided by number of samples
>
> **expectation** is weighted over probabilities of random variable and may or may not be same as empirical mean



### First-Visit MC Policy Evaluation

- Evaluate state s:

  only for first visit to s in every episode:

   - increment counter

     $N(s) = N(s) + 1$

  - increment total return across all episodes

    $S(s) = S(s) + G_t$

  - Value is estimated by mean return over all episodes

    $V(s) = S(s) / N(s)$

Since, by law of large numbers, with sufficient iteration mean tends to actual expectation

### Every-Visit MC Policy Evaluation

It is same as First-Visit MC with a difference that now the states are evaluated for every visit in an episode.

> Comparison if one technique is better than the other is domain dependent.



#### Incremental Mean

This is just to show that mean can be calculated incrementally and need not be done only at the end.

$$
\mu_k = \frac{1}{K} \sum_{j=1}^{k} x_j
$$

$$
= \frac{1}{K}(x_{k} + \sum_{j=1}^{k-1}x_j)
$$

$$
=\frac{1}{K}(x_k + (k-1)\mu_{k-1})
$$

$$
= \mu_{k-1} + \frac{1}{K}(x_k - \mu_{k-1})
$$

- Update old mean in the direction of error to get new mean

### Incremental Monte Carlo Updates

- Update $V(S)$ incrementally after episode

- For each $S_t$ with return $G_t$ use incremental mean to update value function:

  	$N(S_t) \leftarrow N(S_t) + 1$

  	$V(S_t) \leftarrow V(S_t) + \frac{1}{N(S_t)}(G_t - V(S_t))$

- Remove mean and start forgetting old episodes using $\alpha$:

  	$V(S_t) \leftarrow V(S_t) + \alpha(G_t - V(S_t))$

  - This is applicable in non stationary setup where things keeps changeing like environment or policy
  - $\alpha$ helps in move in the direction of error and not all the way



### Temporal Difference Learning

- learns from incomplete episodes by bootstraping

- In MC we had an actual return $G_t$, in simplest TD Learning (0)  we use estimate of return 

  which consists of two parts:

  - immediate reward: $R_{t+1}$ 
  - reward over rest of trajectory: $ \gamma V(S_{t+1})$

- The update of value function is given by just replacing $G_t$ with estimated reward:

  $V(S_t) \leftarrow V(S_t) + \alpha(R_{t+1} + \gamma V(S_{t+1}) - V(S_t))$

- This estimate is also called TD target:

  $R_{t+1} + \gamma V(S_{t+1})$

- Overall error is called TD error:

  $\delta_{t} = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$

### Advantage / Disadvatages:

- With TD we can learn online at every estimate as it is just an estimate of reward; with MC we have to wait till the end of episode before actual return is known.
- Similarily TD can learn from incomplete sequences; MC can only learn from incomplete sequences.
- TD works in non-terminating environment; MC only works for terminating environment.

### TL;DR:

**Goal**: Learn $V_\pi$ online from experience under policy $\pi$

- In MC:

  - update value $V(S_t)$ towards *actual* goal $G_t$:

    $V(S_t) \leftarrow V(S_t) + \alpha(G_t - V(S_t))$

- In TD(0):

  - update value $V(S_t)$ towards *estimated* return:

    $V(S_t) \leftarrow V(S_t) + \alpha(R_{t+1} + \gamma V(S_{t+1}) - V(S_t))$

### Bias / Variance Tradeoff:

- **High bias in TD:** In MC, $G_t$ is unbiased estimate of $V_\pi(S_t)$ whereas TD Target is biased estimate of $V_\pi(S_t)$ hence causing high bias.
- **High variance in MC:** There is a noise in transitions when we consider MC, it is from all the action, transitions, rewards till the end, hence there is a high variance. In case of TD, this variance is only caused from one action, transition, reward which is the current one, hence lower variance.

- MC has high variance, zero bias, good convergence and simple to understand
- TD has low variance, same bias, doesn't always gaurantee convergence and is sensitive to initial values



### Certainty Equivalence:

- MC converges to solution with minimum mean squared error

  - solution that best fits to the observed returns
  - does not exploit markov property - more suited for non-markov processes

  $$
  \sum_{k=1}^{k}\sum_{t=1}^{T_k}(g_t^k - V(s_t^k))^2
  $$
  where, $k$ is an episode and $t$ is a time-step

- TD(0) converges to the solution of maximum likelihood Markov Model

  - solution to the MDP that best fits the data
  - It tries to approximate the transition probability by just counting transitions based on transition probability

  - exploits markov property: builds a markov model with the data and solves for it
    $$
    \hat{P}_{s,s'}^a = \frac{1}{N(s,a)}\sum\sum1(s_t^k, a_t^k, s_{t+1}^k = s, a, s')
    $$

    $$
    \hat{R}_{s,s'}^a = \frac{1}{N(s,a)}\sum\sum1(s_t^k, a_t^k = s, a) r_t^k
    $$
