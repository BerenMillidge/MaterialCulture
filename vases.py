import pprint
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib
import imageio
from copy import deepcopy
import subprocess

matplotlib.use("TkAgg")

# grid
GRID = [30, 30]
GRID_SIZE = np.prod(GRID)

# motifs
HORZ_LINE = 0
HORZ_LINE_2 = 1
VERT_LINE = 2
DIAG_RIGHT_DOWN = 3
DIAG_RIGHT_UP = 4
NUM_MOTIFS = 5

# observations
PIGMENT = 0
NO_PIGMENT = 1

# focal area
FOCAL_AREA = [3, 3]
FOCAL_SIZE = np.prod(FOCAL_AREA)

# generative model
NUM_STATES = FOCAL_SIZE
NUM_ACTIONS = FOCAL_SIZE
NUM_OBSERVATIONS = 2

# action map
ACTION_MAP = [(-1, -1), (0, -1), (1, -1), (-1, 0), (0, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]
VERT_ACTIONS = [0, 1, 2, 6, 7, 8]
HORIZ_ACTIONS = [0, 3, 6, 2, 5, 8]
OPPOSITE_ACTIONS = list(reversed(range(len(ACTION_MAP))))

# stimuli
STIM_0 = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]

STIM_1 = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [3, 3, 3, 3, 3, 3, 3, 3, 3, 3]]

STIM_2 = [
    [4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    [4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
]


STIM_3 = [
    [4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
]

STIM_0 = np.array(STIM_0)
STIM_1 = np.array(STIM_1)
STIM_2 = np.array(STIM_2)
STIM_3 = np.array(STIM_3)


class AttrDict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


def save_gif(imgs, path, fps=32):
    imageio.mimsave(path, imgs, fps=fps)


class Env(object):
    def __init__(self):
        self.obs_matrix = np.zeros((NUM_OBSERVATIONS, GRID[0], GRID[1]))
        self.motif_matrix = np.zeros((NUM_MOTIFS, GRID[0], GRID[1]))
        self.visit_matrix = np.zeros((GRID[0], GRID[1]))
        self.traj_vec = []
        self.x_pos = None
        self.y_pos = None

    def get_A(self):
        A_1 = np.zeros((NUM_OBSERVATIONS, NUM_STATES))
        A_2 = np.zeros((NUM_STATES, NUM_MOTIFS))
        for s in range(NUM_STATES):
            delta = ACTION_MAP[s]
            A_1[:, s] = self.obs_matrix[:, self.x_pos + delta[0], self.y_pos + delta[1]]
            A_2[s, :] = self.motif_matrix[:, self.x_pos + delta[0], self.y_pos + delta[1]]
        return A_1, A_2

    def reset(self, init_x, init_y):
        self.x_pos = init_x
        self.y_pos = init_y
        return self.observe()

    def step(self, action):
        delta = ACTION_MAP[action]
        # clip 1 away from edge to allow for simple remapping
        self.x_pos = np.clip(self.x_pos + delta[0], 1, GRID[0] - 2)
        self.y_pos = np.clip(self.y_pos + delta[1], 1, GRID[1] - 2)
        self.visit_matrix[self.x_pos, self.y_pos] += 1
        self.traj_vec.append((self.x_pos, self.y_pos))
        return self.observe()

    def observe(self):
        obs_vec = self.obs_matrix[:, self.x_pos, self.y_pos]
        motif_vec = self.motif_matrix[:, self.x_pos, self.y_pos]
        return np.argmax(obs_vec), np.argmax(motif_vec)

    def create_motif_env(self, stim_type):
        self.obs_matrix[PIGMENT, :, :] = 0.0
        self.obs_matrix[NO_PIGMENT, :, :] = 1.0
        num_motifs_x = 9
        if stim_type == 0:
            num_motifs_y = 1
        elif stim_type == 1:
            num_motifs_y = 2
        else:
            num_motifs_y = 4
        curr_y = 1
        for xxx in range(num_motifs_y):
            curr_x = 1
            for yyy in range(num_motifs_x):
                if stim_type == 0:
                    motif = STIM_0[xxx, yyy]
                elif stim_type == 1:
                    motif = STIM_1[xxx, yyy]
                elif stim_type == 2:
                    motif = STIM_2[xxx, yyy]
                elif stim_type == 3:
                    motif = STIM_3[xxx, yyy]

                if motif == HORZ_LINE:
                    curr_x = self.horz_line(curr_x, curr_y)
                elif motif == HORZ_LINE_2:
                    curr_x = self.horz_line_2(curr_x, curr_y)
                elif motif == VERT_LINE:
                    curr_x = self.vert_line(curr_x, curr_y)
                elif motif == DIAG_RIGHT_DOWN:
                    curr_x = self.diag_right_down(curr_x, curr_y)
                elif motif == DIAG_RIGHT_UP:
                    curr_x = self.diag_right_up(curr_x, curr_y)
                else:
                    raise ValueError(f"motif {motif} not supported")
            curr_y += 3

    def horz_line(self, x, y):
        self.motif_matrix[HORZ_LINE, x : x + 3, y] = 1.0
        self.obs_matrix[PIGMENT, x : x + 3, y] = 1.0
        self.obs_matrix[NO_PIGMENT, x : x + 3, y] = 0.0

        self.motif_matrix[HORZ_LINE, x : x + 3, y + 2] = 1.0
        self.obs_matrix[PIGMENT, x : x + 3, y + 2] = 1.0
        self.obs_matrix[NO_PIGMENT, x : x + 3, y + 2] = 0.0
        return x + 3

    def horz_line_2(self, x, y):
        self.motif_matrix[HORZ_LINE_2, x : x + 3, y + 1] = 1.0
        self.obs_matrix[PIGMENT, x : x + 3, y + 1] = 1.0
        self.obs_matrix[NO_PIGMENT, x : x + 3, y + 1] = 0.0
        return x + 3

    def vert_line(self, x, y):
        self.motif_matrix[VERT_LINE, x + 2, y : y + 3] = 1.0
        self.obs_matrix[PIGMENT, x + 2, y : y + 3] = 1.0
        self.obs_matrix[NO_PIGMENT, x : x + 2, y : y + 3] = 0.0
        return x + 3

    def diag_right_down(self, x, y):
        grid = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        for xx in range(3):
            for yy in range(3):
                self.motif_matrix[DIAG_RIGHT_DOWN, x : x + xx, y + yy] = 1.0

                if grid[xx, yy] == 0:
                    self.obs_matrix[NO_PIGMENT, x + xx, y + yy] = 1.0
                    self.obs_matrix[PIGMENT, x + xx, y + yy] = 0.0
                elif grid[xx, yy] == 1:
                    self.obs_matrix[PIGMENT, x + xx, y + yy] = 1.0
                    self.obs_matrix[NO_PIGMENT, x + xx, y + yy] = 0.0
        return x + 3

    def diag_right_up(self, x, y):
        grid = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
        for xx in range(3):
            for yy in range(3):
                self.motif_matrix[DIAG_RIGHT_UP, x : x + xx, y + yy] = 1.0

                if grid[xx, yy] == 0:
                    self.obs_matrix[NO_PIGMENT, x + xx, y + yy] = 1.0
                    self.obs_matrix[PIGMENT, x + xx, y + yy] = 0.0
                elif grid[xx, yy] == 1:
                    self.obs_matrix[PIGMENT, x + xx, y + yy] = 1.0
                    self.obs_matrix[NO_PIGMENT, x + xx, y + yy] = 0.0
        return x + 3

    def square(self, x, y):
        pass
        """
        self.motif_matrix[SQUARE, x : x + 3, y : y + 3] = 1.0
        self.obs_matrix[PIGMENT, x : x + 3, y : y + 3] = 1.0
        self.obs_matrix[NO_PIGMENT, x : x + 3, y : y + 3] = 0.0
        return x + 3
        """

    def cross(self, x, y):
        pass
        """
        self.motif_matrix[CROSS, x : x + 3, y + 1] = 1.0
        self.motif_matrix[CROSS, x + 1, y : y + 3] = 1.0

        self.obs_matrix[PIGMENT, x : x + 3, y + 1] = 1.0
        self.obs_matrix[NO_PIGMENT, x : x + 3, y + 1] = 0.0
        self.obs_matrix[PIGMENT, x + 1, y : y + 3] = 1.0
        self.obs_matrix[NO_PIGMENT, x : x + 1, y : y + 3] = 0.0
        return x + 3
        """

    def diamond(self, x, y):
        pass
        """
        grid = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
        for xx in range(3):
            for yy in range(3):
                self.motif_matrix[DIAMOND, x : x + xx, y + yy] = 1.0

                if grid[xx, yy] == 0:
                    self.obs_matrix[NO_PIGMENT, x + xx, y + yy] = 1.0
                    self.obs_matrix[PIGMENT, x + xx, y + yy] = 0.0
                elif grid[xx, yy] == 1:
                    self.obs_matrix[PIGMENT, x + xx, y + yy] = 1.0
                    self.obs_matrix[NO_PIGMENT, x + xx, y + yy] = 0.0
        return x + 3
        """

    def plot(self, t):
        img = np.zeros((GRID[0], GRID[1]))
        img = img + self.obs_matrix[PIGMENT, :, :]
        fig, ax = plt.subplots()
        ax.imshow(img.T, cmap="gray")
        visit_matrix = deepcopy(self.visit_matrix)
        visit_matrix[visit_matrix == 0] = np.nan
        ax.imshow(visit_matrix.T, alpha=0.7, vmin=0)
        """
        locs_x = []
        locs_y = []
        for i, pos in enumerate(self.traj_vec):
            if i % 4 == 0 and i > 0:
                locs_x.append(pos[0])
                locs_y.append(pos[1])
        plt.plot(locs_x, locs_y, color="red", alpha=0.5)
        """
        ax.scatter([self.x_pos], [self.y_pos], color="red", s=5)
        ax.set_title(f"Step: {t}")
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()
        return img

    def plot_2(self, t, motif):
        fig, axes = plt.subplots(2, 2)

        """ top left """
        img = np.zeros((GRID[0], GRID[1]))
        img = img + self.obs_matrix[PIGMENT, :, :]
        axes[0, 0].imshow(img.T, cmap="gray")
        visit_matrix = deepcopy(self.visit_matrix)
        visit_matrix[visit_matrix == 0] = np.nan
        axes[0, 0].imshow(visit_matrix.T, alpha=0.7, vmin=0)
        axes[0, 0].scatter([self.x_pos], [self.y_pos], color="red", s=5)
        axes[0, 0].set_title(f"Step: {t}")
        axes[0, 0].get_xaxis().set_visible(False)
        axes[0, 0].get_yaxis().set_visible(False)

        """ top right """
        if motif == HORZ_LINE:
            grid = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]]) 
        elif motif == HORZ_LINE_2:
            grid = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]]) 
        elif motif == VERT_LINE:
            grid = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]])
        elif motif == DIAG_RIGHT_DOWN:
            grid = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        elif motif == DIAG_RIGHT_UP:
            grid = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
        else:
            raise ValueError(f"motif {motif} not supported")

        axes[0, 1].imshow(grid, cmap="gray")
        axes[0, 1].get_xaxis().set_visible(False)
        axes[0, 1].get_yaxis().set_visible(False)
        axes[0, 1].set_title("Inferred motif")

        """ bottom left """
        locs_x = []
        locs_y = []
        for i, pos in enumerate(self.traj_vec):
            if i % 4 == 0 and i > 0:
                locs_x.append(GRID[0] - pos[0])
                locs_y.append(pos[1])
        
        axes[1, 1].set_xlim(GRID[0])
        axes[1, 1].set_ylim(GRID[1])
        axes[1, 1].plot(locs_x, locs_y, '-o', color="red", alpha=0.4)
        axes[1, 1].get_xaxis().set_visible(False)
        axes[1, 1].get_yaxis().set_visible(False)
        axes[1, 1].set_title("Scan path")

        """ bottom right """
        axes[1, 0].imshow(visit_matrix.T, alpha=0.7, vmin=0)
        axes[1, 0].get_xaxis().set_visible(False)
        axes[1, 0].get_yaxis().set_visible(False)
        axes[1, 0].set_title("Heatmap")


        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()
        return img


class MDP(object):
    def __init__(self, A, B, A_2, B_2, C, level_two_tick=1, prev_position_prior=False):
        self.level_two_tick = level_two_tick
        self.prev_position_prior = prev_position_prior
        self.A = A
        self.B = B
        self.A_2 = A_2
        self.B_2 = B_2
        self.C = C
        self.p0 = np.exp(-16)

        self.num_states = self.A.shape[1]
        self.num_obs = self.A.shape[0]
        self.num_actions = self.B.shape[0]
        self.num_motifs = self.B_2.shape[0]

        self.A = self.A + self.p0
        self.A = self.normdist(self.A)
        self.lnA = np.log(self.A)

        self.A_2 = self.A_2 + self.p0
        self.A_2 = self.normdist(self.A_2)
        self.lnA_2 = np.log(self.A_2)

        self.B = self.B + self.p0
        for a in range(self.num_actions):
            self.B[a] = self.normdist(self.B[a])

        self.B_2 = self.B_2 + self.p0
        self.B_2 = self.normdist(self.B_2)
        self.B_a_2 = deepcopy(self.B_2)

        self.C = self.C + self.p0
        self.C = self.normdist(self.C)

        self.sQ = np.zeros([self.num_states, 1])
        self.sQ_2 = np.zeros([self.num_motifs, 1])
        self.aQ = np.zeros([self.num_actions, 1])
        self.prev_action = None
        self.prev_motif = 0
        self.t = 0

    def set_A(self, A, A_2):
        self.A = A + self.p0
        self.A = self.normdist(self.A)
        self.lnA = np.log(self.A)

        self.A_2 = A_2 + self.p0
        self.A_2 = self.normdist(self.A_2)
        self.lnA_2 = np.log(self.A_2)

    def reset(self, obs):
        self.t = 0
        self.curr_obs = obs
        likelihood = self.lnA[obs, :]
        likelihood = likelihood[:, np.newaxis]
        self.sQ = self.softmax(likelihood)
        self.prev_action = self.random_action()

    def step(self, obs, motif_obs, sample=True):
        """ state inference """
        likelihood = self.lnA[obs, :]
        likelihood = likelihood[:, np.newaxis]
        prior = np.dot(self.B[self.prev_action], self.sQ)
        prior = np.log(prior)
        # ignore likelihood for now
        # self.sQ = self.softmax(likelihood + prior)
        self.sQ = self.softmax(prior)

        likelihood_2 = np.dot(self.lnA_2.T, self.sQ)
        prior_2 = np.dot(self.B_2, self.sQ_2)
        prior_2 = np.log(prior_2 + self.p0)
        self.sQ_2 = self.softmax(likelihood_2 + prior_2)

        """ action inference """
        neg_efe = np.zeros([self.num_actions, 1])
        for a in range(self.num_actions):
            fs = np.dot(self.B[a], self.sQ)
            fo = np.dot(self.A, fs)
            fo = self.normdist(fo + self.p0)
            utility = np.sum(fo * np.log(fo / self.C), axis=0)
            utility = utility[0]
            neg_efe[a] -= utility

        # stay prior
        neg_efe[4] -= 20.0

        if self.prev_position_prior:
            # previous location prior
            neg_efe[OPPOSITE_ACTIONS[self.prev_action]] -= 10.0

        # action selection
        self.uQ = self.softmax(neg_efe)

        if sample:
            action = np.argmax(np.random.multinomial(1, self.uQ))
        else:
            # selection maximum
            hu = max(self.uQ)
            options = np.where(self.uQ == hu)[0]
            action = int(np.random.choice(options))
        self.prev_action = action

        # TODO use sQ_2
        if self.t % self.level_two_tick == 0:
            self.B_a_2[motif_obs, self.prev_motif] += 0.5
            self.B_2 = self.normdist(np.copy(self.B_a_2))
            self.prev_motif = motif_obs
            self.t = 0
        return action

    def random_action(self):
        return int(np.random.choice(range(self.num_actions)))

    @staticmethod
    def softmax(x):
        x = x - x.max()
        x = np.exp(x)
        x = x / np.sum(x)
        return x

    @staticmethod
    def normdist(x):
        return np.dot(x, np.diag(1 / np.sum(x, 0)))


def main(cf):
    A_1 = np.zeros((NUM_OBSERVATIONS, NUM_STATES))
    A_2 = np.zeros((NUM_STATES, NUM_MOTIFS))

    B_1 = np.zeros((NUM_ACTIONS, NUM_STATES, NUM_STATES))
    for a in range(NUM_ACTIONS):
        B_1[a, a, :] = 1.0
    B_2 = np.eye(NUM_MOTIFS)

    C = np.zeros((NUM_OBSERVATIONS, 1))
    C[PIGMENT] = 0.9
    C[NO_PIGMENT] = 0.1

    mdp = MDP(A_1, B_1, A_2, B_2, C, level_two_tick=cf.level_two_tick, prev_position_prior=cf.prev_position_prior)
    env = Env()

    env.create_motif_env(cf.stim_type)
    init_x = GRID[0] // 2
    init_y = GRID[1] // 2
    obs, motif_obs = env.reset(init_x=init_x, init_y=init_y)
    curr_motif = motif_obs 
    A_1, A_2 = env.get_A()
    mdp.set_A(A_1, A_2)
    mdp.reset(obs)

    imgs = []
    nb_vert = 0
    nb_horiz = 0
    for t in range(cf.num_steps):
        A_1, A_2 = env.get_A()
        mdp.set_A(A_1, A_2)
        action = mdp.step(obs, motif_obs)
        obs, motif_obs = env.step(action)

        if t % 4 == 0:
            curr_motif = motif_obs

        img = env.plot_2(t, curr_motif)
        imgs.append(img)
        if action in VERT_ACTIONS:
            nb_vert = nb_vert + 1
        if action in HORIZ_ACTIONS:
            nb_horiz = nb_horiz + 1

    vert_index = (GRID[1] * nb_vert - GRID[0] * nb_horiz) / (GRID[1] * nb_vert + GRID[1] * nb_vert)
    print(f"vert_index: {vert_index}")
    if cf.save:
        np.save(cf.base_name + "/visit_matrix_" + str(cf.stim_type) + ".npy", env.visit_matrix)
        np.save(cf.base_name + "/trajectory_vector_" + str(cf.stim_type) + ".npy", np.array(env.traj_vec))

    save_gif(imgs, cf.gif_path)
    return vert_index, mdp.B_2, img

if __name __ == '__main__':
    cf = AttrDict()
    cf.num_steps = 100
    vertical_indices = []
    cf.save = True
    cf.level_two_tick = 4
    cf.base_name = "single_row_2"
    cf.prev_position_prior = True
    B_matrices = []
    entropies = []
    imgs = []
    subprocess.call(["mkdir", "-p", str(cf.base_name)])
    for i in range(4):
        cf.stim_type = i
        # adjust grid sizes
        if cf.stim_type == 0:
            GRID = (30, 5)
        elif cf.stim_type == 1:
            GRID = (30, 8)
        elif cf.stim_type == 3:
            GRID = (30, 14)
        else:
            GRID = (30, 14)
        GRID_SIZE = np.prod(GRID)

        cf.gif_path = cf.base_name + "/test_" + str(i) + ".gif"
        vi, B, img = main(cf)
        np.save(cf.base_name + "/B_matrix_" + str(i) + ".npy", B)
        vertical_indices.append(vi)
        B_matrices.append(B)
        entropy = []
        for m in range(B.shape[1]):
            entropy.append(stats.entropy(B[:, m]))
        entropies.append(sum(entropy))
        imgs.append(img)

    print("vertical_indices: ", vertical_indices)
    print("entropies: ", entropies)
    pprint.pprint(np.round(B_matrices, 3))
    vertical_indices = np.array(vertical_indices)
    np.save(cf.base_name + "/vertical_indices.npy", vertical_indices)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].plot(vertical_indices, color="red")
    axes[0].set_ylabel("Vertical Index (VI)")
    axes[0].set_xlabel("Artefact Complexity")
    axes[1].plot(entropies, color="green")
    axes[1].set_ylabel("C-PAST (transition entropy)")
    axes[1].set_xlabel("Artefact Complexity")
    plt.tight_layout()
    plt.savefig(cf.base_name + "/figure.pdf", dpi=300)

    fig, ax = plt.subplots(1, 1)
    ax.imshow(imgs[0])
    plt.axis("off")
    plt.savefig(cf.base_name + "/figure_1.pdf", dpi=300)
    fig, ax = plt.subplots(1, 1)
    ax.imshow(imgs[1])
    plt.axis("off")
    plt.savefig(cf.base_name + "/figure_2.pdf", dpi=300)
    fig, ax = plt.subplots(1, 1)
    ax.imshow(imgs[2])
    plt.axis("off")
    plt.savefig(cf.base_name + "/figure_3.pdf", dpi=300)
    fig, ax = plt.subplots(1, 1)
    ax.imshow(imgs[3])
    plt.axis("off")
    plt.savefig(cf.base_name + "/figure_4.pdf", dpi=300)