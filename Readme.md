This code is being developed to first replicate the results of "Learning Attentional Communication for Multiagent Systems" paper.
Once that is done, I will be looking towards extending this work.

The directories are structured in the same way as MADDPG repository.

General Notes:
- MADDPG uses a separate centralized critic for each agent, and a separate decentralized actor for each agent.
- In ATOC, all agents share the parameters of the critic, policy network, attention unit, and communication channel.
    Also, I think ATOC uses local observations for Q updates. Please check. Yes, it uses local observations.
- In COMA, all agents have a single centralized critic and a single actor policy used by all agents.
- The MADDPG's agents use global q-function. That's the reason why the obs_shape_n and act_space_n
for each of the agents is a list so that it can have placeholders for all the observations and actions for all the other agents.
That is how MADDPG gets across the problem of partial observability and changing environment when updating the parameters. By having
the actions of other agents as well, the environment is no longer non-stationary. However, we want to use local q-function.
So if we use "ddpg" in the arglist instead of "maddpg" then it will use local q-function.


Some useful env properties (see environment.py):
.observation_space  :   Returns the observation space for each agent
.action_space       :   Returns the action space for each agent
.n                  :   Returns the number of Agents



IMPLEMENTATION LOG:
- added replay buffer
- Initialized the actor and the critic network. This seems to work fine with the code that was already present for the maddpg.
However I had to change the args.good_policy and args.bad_policy from to maddpg to ddpg since maddpg is a global q function,
and rather than global q-function, we want a local q-function since ATOC and COMA are take local observations.

- In the "Recurrent Models of Visual Attention" paper, the attention network is defined in more detail in the "Experiments" section.


TODO:
- Initialize critic network (get it done today)
- Initialize actor network (get it done today)
- Initialize communication channel (get it done today)
- Initialize attention classifier (get it done today)
- Initialize target networks (get it done today)
- Initialize queue D (get it done today)