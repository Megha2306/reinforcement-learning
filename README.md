# reinforcement-learning

CIA 1:  K-arm bandit based solution for A recommendation system.

Challenge: Cold start problem

Problem Statement
The aim is to build a real-time recommendation system that addresses the cold start problem. When new items are introduced or when new users interact with the system, traditional recommendation algorithms may lack sufficient data to make relevant suggestions, impacting engagement rates. This challenge can be tackled using a K-arm bandit approach, specifically Thompson Sampling combined with content-based grouping. The primary objective is to improve recommendations for new items or new users by leveraging content similarities and Bayesian learning.

Objective
Maximize user engagement by:

Effectively recommending items in the absence of historical interaction data.
Balancing exploration of new items with exploitation of popular or proven items in each content category.
Algorithm Design
1. Initialization with Content-Based Clustering
Each content item belongs to a content group (or cluster) defined by its metadata (such as genre, tags, or type). For example, movies could be clustered by genres like "action" or "comedy." These groups allow the model to leverage similarities for new items, reducing the need for initial interaction data.

Let:

𝐺
G denote the set of content groups: 
𝐺
=
{
𝑔
1
,
𝑔
2
,
…
,
𝑔
𝑚
}
G={g 
1
​
 ,g 
2
​
 ,…,g 
m
​
 }.
𝐾
𝑖
K 
i
​
  be the number of items in group 
𝑔
𝑖
g 
i
​
 .
Each group has a Thompson Sampling-based reward distribution associated with each item. We model the probability of engagement for an item 
𝑘
k in group 
𝑔
g as a Bernoulli distribution parameterized by 
𝑝
𝑘
p 
k
​
 , where 
𝑝
𝑘
p 
k
​
  is unknown.

2. Thompson Sampling for Recommendation
For each item 
𝑘
k in group 
𝑔
g:

We estimate the success rate of 
𝑝
𝑘
p 
k
​
  using a Beta distribution prior 
Beta
(
𝛼
𝑘
,
𝛽
𝑘
)
Beta(α 
k
​
 ,β 
k
​
 ), where:
𝛼
𝑘
α 
k
​
  represents the number of "successes" (positive engagements).
𝛽
𝑘
β 
k
​
  represents the number of "failures" (negative engagements).
Initially, for all items, set 
𝛼
𝑘
=
1
α 
k
​
 =1 and 
𝛽
𝑘
=
1
β 
k
​
 =1 (uninformative prior), meaning we have no initial data.

3. Update Rule
After each recommendation and user response:

If the user engages with item 
𝑘
k (e.g., clicks), update 
𝛼
𝑘
←
𝛼
𝑘
+
1
α 
k
​
 ←α 
k
​
 +1.
If the user does not engage, update 
𝛽
𝑘
←
𝛽
𝑘
+
1
β 
k
​
 ←β 
k
​
 +1.
4. Decision-Making: Sampling from Beta Distribution
For each user interaction:

For each item 
𝑘
k in the selected group 
𝑔
g, sample a reward estimate 
𝑝
^
𝑘
p
^
​
  
k
​
  from the distribution 
Beta
(
𝛼
𝑘
,
𝛽
𝑘
)
Beta(α 
k
​
 ,β 
k
​
 ).
Select the item with the highest sampled reward 
𝑝
^
𝑘
p
^
​
  
k
​
  to recommend.
Mathematical Formulation
Item Reward Sampling:

𝑝
^
𝑘
∼
Beta
(
𝛼
𝑘
,
𝛽
𝑘
)
p
^
​
  
k
​
 ∼Beta(α 
k
​
 ,β 
k
​
 )
where 
𝛼
𝑘
α 
k
​
  and 
𝛽
𝑘
β 
k
​
  are updated with each user engagement.

Expected Reward Update:

For each engagement (click):
𝛼
𝑘
=
𝛼
𝑘
+
1
α 
k
​
 =α 
k
​
 +1
For each non-engagement (no click):
𝛽
𝑘
=
𝛽
𝑘
+
1
β 
k
​
 =β 
k
​
 +1
Maximizing Expected Reward:

Select item 
𝑘
k that maximizes 
𝑝
^
𝑘
p
^
​
  
k
​
  from the Beta distribution.


CIA 2: Create a 100x100 grid with obstacles in between 2 random points. Build an MDP based RL agent to optimise both policies and actions at every state. Benchmark DP method with other RL solutions for the same problem.

Explanation of Key Steps:
Grid Setup:

Rewards: Regular states have a small negative reward to encourage the agent to reach the goal quickly. The goal state has a positive reward.
Obstacles: Randomly placed obstacles with negative rewards discourage the agent from passing through these points.
Action Selection:

The agent chooses between exploring randomly and exploiting the best-known action based on the epsilon-greedy strategy.
Q-learning Update:

The Q-value update formula uses the Q-learning rule, incorporating the learning rate (alpha), discount factor (gamma), and the maximum Q-value for the next state.
Stopping Condition:

Each episode runs until the agent reaches the goal state at (99, 99)

Why Q-Learning for This Problem?
Focus on Optimizing to the Goal:

In our grid world, the primary objective is to find the most direct path to the goal while avoiding obstacles, which makes an aggressive approach like Q-Learning more effective. Q-Learning will converge faster toward an optimal policy that maximizes rewards, giving it an edge for this particular task.
More Efficient Exploration:

Since our grid is large (100x100), Q-Learning’s bias toward optimal actions accelerates the learning process, making it more efficient in terms of convergence time. SARSA would take longer, as it updates based on actions it actually takes, including exploratory steps that might not be optimal.
