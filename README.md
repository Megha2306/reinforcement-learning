# reinforcement-learning

# CIA 1:  K-arm bandit based solution for A recommendation system.

# Challenge: Cold start problem

### **Problem Statement**

The aim is to build a real-time recommendation system that addresses the cold start problem. When new items are introduced or when new users interact with the system, traditional recommendation algorithms may lack sufficient data to make relevant suggestions, impacting engagement rates. This challenge can be tackled using a K-arm bandit approach, specifically Thompson Sampling combined with content-based grouping. The primary objective is to improve recommendations for new items or new users by leveraging content similarities and Bayesian learning.

Objective

Maximize user engagement by:
1. Effectively recommending items in the absence of historical interaction data.
2. Balancing exploration of new items with exploitation of popular or proven items in each content category.

Algorithm Design

1. Initialization with Content-Based Clustering

Each content item belongs to a content group (or cluster) defined by its metadata (such as genre, tags, or type). For example, movies could be clustered by genres like "action" or "comedy." These groups allow the model to leverage similarities for new items, reducing the need for initial interaction data.

Let:
- \( G \) denote the set of content groups: \( G = \{g_1, g_2, \dots, g_m\} \).
- \( K_i \) be the number of items in group \( g_i \).

Each group has a Thompson Sampling-based reward distribution associated with each item. We model the probability of engagement for an item \( k \) in group \( g \) as a **Bernoulli distribution** parameterized by \( p_k \), where \( p_k \) is unknown.

2. Thompson Sampling for Recommendation

For each item \( k \) in group \( g \):
- We estimate the success rate of \( p_k \) using a **Beta distribution** prior \( \text{Beta}(\alpha_k, \beta_k) \), where:
  - \( \alpha_k \) represents the number of "successes" (positive engagements).
  - \( \beta_k \) represents the number of "failures" (negative engagements).

Initially, for all items, set \( \alpha_k = 1 \) and \( \beta_k = 1 \) (uninformative prior), meaning we have no initial data.

3. Update Rule

After each recommendation and user response:
1. If the user engages with item \( k \) (e.g., clicks), update \( \alpha_k \leftarrow \alpha_k + 1 \).
2. If the user does not engage, update \( \beta_k \leftarrow \beta_k + 1 \).

4. Decision-Making: Sampling from Beta Distribution

For each user interaction:
1. For each item \( k \) in the selected group \( g \), sample a reward estimate \( \hat{p}_k \) from the distribution \( \text{Beta}(\alpha_k, \beta_k) \).
2. Select the item with the highest sampled reward \( \hat{p}_k \) to recommend.

---

Mathematical Formulation

1. Item Reward Sampling:
   \[
   \hat{p}_k \sim \text{Beta}(\alpha_k, \beta_k)
   \]
   where \( \alpha_k \) and \( \beta_k \) are updated with each user engagement.

2. Expected Reward Update:
   - For each engagement (click):
     \[
     \alpha_k = \alpha_k + 1
     \]
   - For each non-engagement (no click):
     \[
     \beta_k = \beta_k + 1
     \]

3. Maximizing Expected Reward:
   - Select item \( k \) that maximizes \( \hat{p}_k \) from the Beta distribution.



# CIA 2: Create a 100x100 grid with obstacles in between 2 random points. Build an MDP based RL agent to optimise both policies and actions at every state. Benchmark DP method with other RL solutions for the same problem.

# Explanation of Key Steps:
Grid Setup:

Rewards: Regular states have a small negative reward to encourage the agent to reach the goal quickly. The goal state has a positive reward.
Obstacles: Randomly placed obstacles with negative rewards discourage the agent from passing through these points.
Action Selection:

The agent chooses between exploring randomly and exploiting the best-known action based on the epsilon-greedy strategy.
Q-learning Update:

The Q-value update formula uses the Q-learning rule, incorporating the learning rate (alpha), discount factor (gamma), and the maximum Q-value for the next state.
Stopping Condition:

Each episode runs until the agent reaches the goal state at (99, 99)

# Why Q-Learning for This Problem?
Focus on Optimizing to the Goal:

In our grid world, the primary objective is to find the most direct path to the goal while avoiding obstacles, which makes an aggressive approach like Q-Learning more effective. Q-Learning will converge faster toward an optimal policy that maximizes rewards, giving it an edge for this particular task.
More Efficient Exploration:

Since our grid is large (100x100), Q-Learning’s bias toward optimal actions accelerates the learning process, making it more efficient in terms of convergence time. SARSA would take longer, as it updates based on actions it actually takes, including exploratory steps that might not be optimal.
