### Experiments to run:
---
- BiLSTM
	- understand experimental setup
	- experiment with its loss, its parameters and experiment setup
	- scale up with **CIFAR-10** see how it works
    - do interpretability tests to understand exactly what its pruning
---

- Activation Pruning (+BiLSTM)
	- compare with activation pruning (lamdba sweep)
    - also do similar chaining, then interpretability tests
---

- RL algorithm decision
    - work on PPO (entropy bonus tweaking), actor critic, and fin tune reinforce
    - set up three basic algorithms to run on all environmental setup
    - get SOTA, and run state/reward experiments

- RL Experiments
    - properly understand experimental setup
    - tweak state 
        - (local state + entropy bonus exploration?)
        - (local state + ICM?)
        - (entire model itself?)
    - what if to make policy and value net permutation invariant we add attention in?
    - tweak environmental setup to see if we can run an episode forever?? and just keep max pruning with an acceptable accuracy drop level
    


### RL Experiments

Current MDP:
State -> 

per alive neuron (5):
[
    incoming weights (abs) then summed
    incoming sqrt(sum(squared))
    ouitgoing weights (abs) then summed
    outgoing sqrt(sum(squared))
    mean actvation
]

per neuron -> one hot vector for which hidden linear
        -> column vector, each scalar being percentage of layer pruned
            *slight redundancy, because this is repeated globally*

global stats
    1. change in accuracy (global, not from this step)
        makes it treat it like a path? should that happen or no?
        if we give it global, it should net make no real difference
        **cant give it global straight up, would make it non markovian**
    2. fraction pruned so far
    3. alive_map (neurons, layers) which are alive and which arent



Action ->

alive_map is built from alive neurons each observation. 
each alive neuron in order is assigned its neuron, layer pair. 
basically idx n will contain neuron, layer for (n-1)th alive neuron

action is in the form of indices
for newly dead neurons
16 neurons killed each time (note this weirds out log prob calculation. )
    -> find a repr where 16 is considered 1 action
    -> increases variance because we now have to find the log prob of the categorical distribution that varied this


Reward ->

simple delta in accuracy (reward is a part of the state??)
