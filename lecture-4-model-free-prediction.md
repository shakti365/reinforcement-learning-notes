# Lecture 4: Model Free Prediction

**Model free prediction:**

- Estimate the value function of an unknown MDP when policy is given



### Monte Carle Reinforcement Learning

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



> **empirical mean** is the total of samples divided by number of samples
>
> **expectation** is weighted over probabilities of random variable and may or may not be same as empirical mean



### First-Visit MC Policy Evaluation

- Evaluate state s:

  only for first visit to s in every episode:

   - increment counter

     $N(s) = N(s) + 1$

  - increment total return across all episodes

    $S(s) = S(s) + G_t$

  Value us estimated by mean return over all episodes

  $V(s) = S(s) / N(s)$

Since, by law of large numbers, with sufficient iteration mean tends to actual expectation

