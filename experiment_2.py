#Code to run experiment 2 of the paper which implements a psychophysics tasks in which the agent must select the correct
#motif to fill in a cut out and simulated hits are recorded.

import pprint
import numpy as np
import matplotlib.pyplot as plt
​
class MDP(object):
    def __init__(self, A, B):
        self.A = A
        self.B = B
        self.p0 = np.exp(-16)
        self.A = self.A + self.p0
        self.A = self.normdist(self.A)
        self.lnA = np.log(self.A)
        self.B = self.B + self.p0
        self.B = self.normdist(self.B)
        self.num_states = self.A.shape[1]
        self.num_obs = self.A.shape[0]
        self.sQ = np.zeros([self.num_states, 1])
        
    def reset(self, obs):
        likelihood = self.lnA[obs, :]
        likelihood = likelihood[:, np.newaxis]
        self.sQ = self.softmax(likelihood)
​
    def infer(self, obs):
        """ state inference """
        likelihood = self.lnA[obs, :]
        likelihood = likelihood[:, np.newaxis]
        prior = np.dot(self.B, self.sQ)
        prior = np.log(prior)
        self.sQ = self.softmax(likelihood + prior)
        return self.sQ
​
    @staticmethod
    def softmax(x):
        x = x - x.max()
        x = np.exp(x)
        x = x / np.sum(x)
        return x
​
    @staticmethod
    def normdist(x):
        return np.dot(x, np.diag(1 / np.sum(x, 0)))
​
​
​
path = "single_row/B_matrix_{}.npy"
num_motifs = 5
num_conditions = 4
​
trial_struct = [4, 0, 3, 2, 0] * 20
num_trials = len(trial_struct)
​
hits = []
num_hits = []
for condition in range(num_conditions):
    hits.append([])
    num_hits.append(0)
    condition_path = path.format(condition)
    A = np.eye(num_motifs) 
    B = np.load(condition_path)
    mdp = MDP(A, B)
    mdp.reset(0)
    for cut_out in trial_struct:
        sQ = mdp.infer(cut_out)
        state = np.argmax(np.random.multinomial(1, sQ))
        if state == cut_out:
            hits[condition].append(1)
            num_hits[condition] += 1
        else:
            hits[condition].append(0)
    print(f"condition {condition} / num_hits {num_hits[condition]})")
​
fig, axes = plt.subplots(1, 5, figsize=(12, 4))
for condition in range(num_conditions):
    axes[condition].scatter(range(num_trials), hits[condition])
    axes[condition].set_ylim((-0.2, 1.2))
    axes[condition].set_xlabel("Trial")
    axes[condition].set_ylabel("Hit")
    axes[condition].set_title(f"Artefact Complexity {condition}")
​
axes[-1].plot(num_hits, color="red")
axes[-1].set_xlabel("Artefact Complexity")
axes[-1].set_ylabel("Hit Rate")
axes[-1].set_title(f"Artefact Complexity / Hit Rate")
plt.tight_layout()
plt.show()