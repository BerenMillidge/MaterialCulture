import pprint
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import imageio
from copy import deepcopy

matplotlib.use('TkAgg')

GRID = [30, 30]
GRID_SIZE = np.prod(GRID)

"""
GET RID OF SPARE MOTIFS
"""

HORZ_LINE = 0
HORZ_LINE_2 = 1
VERT_LINE = 2
DIAG_RIGHT_DOWN = 3
DIAG_RIGHT_UP = 4
NUM_MOTIFS = 5

FOCAL_AREA = [3, 3]
FOCAL_SIZE = np.prod(FOCAL_AREA)

NUM_STATES = FOCAL_SIZE
NUM_ACTIONS = FOCAL_SIZE
NUM_OBSERVATIONS = 2

PIGMENT = 0
NO_PIGMENT = 1

ACTION_MAP = [(-1, -1), (0, -1), (1, -1),
              (-1,  0), (0,  0), (1,  0),
              (-1,  1), (0,  1), (1,  1)]

STIM_0 = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]

STIM_1 = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [3, 3, 3, 3, 3, 3, 3, 3, 3, 3]]

STIM_2 = [[4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
          [4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
          [4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
          [4, 4, 4, 4, 4, 4, 4, 4, 4, 4]]


STIM_3 = [[4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
          [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
          [4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
          [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
          [4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],]

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
        self.trajectory_vector = []
        self.x_pos = None
        self.y_pos = None

    def get_A(self):
        A = np.zeros((NUM_OBSERVATIONS, NUM_STATES))
        A_2 = np.zeros((NUM_STATES, NUM_MOTIFS))
        for s in range(NUM_STATES):
            delta = ACTION_MAP[s]
            A[:, s] = self.obs_matrix[:, self.x_pos + delta[0], self.y_pos + delta[1]]
            A_2[s, :] = self.motif_matrix[:, self.x_pos + delta[0], self.y_pos + delta[1]]
        return A, A_2

    def reset(self, init_x, init_y):
        self.x_pos = init_x
        self.y_pos = init_y
        return self.observe()

    def step(self, action):
        delta = ACTION_MAP[action]
        # clip 1 away from edge to allow for simple remapping
        self.x_pos = np.clip(self.x_pos + delta[0], 1, GRID[0] - 2)
        self.y_pos = np.clip(self.y_pos + delta[1], 1, GRID[1] - 2)
        self.visit_matrix[self.x_pos, self.y_pos] += 1
        self.trajectory_vector.append((self.x_pos, self.y_pos))
        return self.observe()
        
    def observe(self):
        obs_vec = self.obs_matrix[:, self.x_pos, self.y_pos]
        motif_vec = self.motif_matrix[:, self.x_pos, self.y_pos]
        return np.argmax(obs_vec), np.argmax(motif_vec)

    def create_motif_env(self, stim_type):
        def diag_right_down(x, y):
            grid = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            for xx in range(3):
                for yy in range(3):
                    self.motif_matrix[DIAG_RIGHT_DOWN, x: x + xx, y + yy] = 1.0

                    if grid[xx, yy] == 0:
                        self.obs_matrix[NO_PIGMENT, x + xx, y + yy] = 1.0
                        self.obs_matrix[PIGMENT, x + xx, y + yy] = 0.0
                    elif grid[xx, yy] == 1:
                        self.obs_matrix[PIGMENT, x + xx, y + yy] = 1.0
                        self.obs_matrix[NO_PIGMENT, x + xx, y + yy] = 0.0
            return x + 3

        def diag_right_up(x, y):
            grid = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
            for xx in range(3):
                for yy in range(3):
                    self.motif_matrix[DIAG_RIGHT_UP, x: x + xx, y + yy] = 1.0

                    if grid[xx, yy] == 0:
                        self.obs_matrix[NO_PIGMENT, x + xx, y + yy] = 1.0
                        self.obs_matrix[PIGMENT, x + xx, y + yy] = 0.0
                    elif grid[xx, yy] == 1:
                        self.obs_matrix[PIGMENT, x + xx, y + yy] = 1.0
                        self.obs_matrix[NO_PIGMENT, x + xx, y + yy] = 0.0
            return x + 3

        def square(x, y):    
            """ 
            Should be +2 rather than +3???
            """   
            self.motif_matrix[SQUARE, x: x + 3, y: y + 3] = 1.0
            self.obs_matrix[PIGMENT, x: x + 3, y: y + 3] = 1.0
            self.obs_matrix[NO_PIGMENT, x: x + 3, y: y + 3] = 0.0
            return x + 3
            
        def horz_line(x, y):
            self.motif_matrix[HORZ_LINE, x: x + 3, y] = 1.0
            self.obs_matrix[PIGMENT, x: x + 3, y] = 1.0
            self.obs_matrix[NO_PIGMENT, x: x + 3, y] = 0.0

            self.motif_matrix[HORZ_LINE, x: x + 3, y + 2] = 1.0
            self.obs_matrix[PIGMENT, x: x + 3, y + 2] = 1.0
            self.obs_matrix[NO_PIGMENT, x: x + 3, y + 2] = 0.0
            return x + 3

        def horz_line_2(x, y):
            self.motif_matrix[HORZ_LINE_2, x: x + 3, y+1] = 1.0
            self.obs_matrix[PIGMENT, x: x + 3, y+1] = 1.0
            self.obs_matrix[NO_PIGMENT, x: x + 3, y+1] = 0.0
            return x + 3

        def vert_line(x, y):
            self.motif_matrix[VERT_LINE, x + 2, y: y + 3] = 1.0
            self.obs_matrix[PIGMENT, x + 2, y: y + 3] = 1.0
            self.obs_matrix[NO_PIGMENT, x: x + 2, y: y + 3] = 0.0
            return x + 3

        def cross(x, y):
            self.motif_matrix[CROSS, x: x + 3, y + 1] = 1.0
            self.motif_matrix[CROSS, x + 1, y: y + 3] = 1.0
            
            self.obs_matrix[PIGMENT, x: x + 3, y + 1] = 1.0
            self.obs_matrix[NO_PIGMENT, x: x + 3, y + 1] = 0.0
            self.obs_matrix[PIGMENT, x + 1, y: y + 3] = 1.0
            self.obs_matrix[NO_PIGMENT, x: x + 1, y: y + 3] = 0.0
            return x + 3

        def diamond(x, y):
            grid = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
            for xx in range(3):
                for yy in range(3):
                    self.motif_matrix[DIAMOND, x: x + xx, y + yy] = 1.0

                    if grid[xx, yy] == 0:
                        self.obs_matrix[NO_PIGMENT, x + xx, y + yy] = 1.0
                        self.obs_matrix[PIGMENT, x + xx, y + yy] = 0.0
                    elif grid[xx, yy] == 1:
                        self.obs_matrix[PIGMENT, x + xx, y + yy] = 1.0
                        self.obs_matrix[NO_PIGMENT, x + xx, y + yy] = 0.0
            return x + 3

        self.obs_matrix[PIGMENT, :, :] = 0.0
        self.obs_matrix[NO_PIGMENT, :, :] = 1.0
        num_motifs_x = (GRID[0] // 3) - 1
        num_motifs_y = (GRID[1] // 3) - 1
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
                    curr_x = horz_line(curr_x, curr_y)
                elif motif == HORZ_LINE_2:
                    curr_x = horz_line_2(curr_x, curr_y)
                elif motif == VERT_LINE:
                    curr_x = vert_line(curr_x, curr_y)
                elif motif == DIAG_RIGHT_DOWN:
                    curr_x = diag_right_down(curr_x, curr_y)
                elif motif == DIAG_RIGHT_UP:
                    curr_x = diag_right_up(curr_x, curr_y)
                else:
                    raise ValueError(f"motif {motif} not supported")
            curr_y += 3

    def create_random_env(self, num_pixels):
        self.obs_matrix[PIGMENT, :, :] = 0.0
        self.obs_matrix[NO_PIGMENT, :, :] = 1.0
        curr_x = GRID[0] // 2
        curr_y = GRID[1] // 2 
        self.obs_matrix[PIGMENT, curr_x, curr_y] = 1.0
        self.obs_matrix[NO_PIGMENT, curr_x, curr_y] = 0.0
        for _ in range(num_pixels):
            delta_x = np.random.choice([-1, 0, 1])
            delta_y = np.random.choice([-1, 0, 1])
            curr_x = np.clip(curr_x + delta_x, 1, GRID[0] - 1)
            curr_y = np.clip(curr_y + delta_y, 1, GRID[1] - 1)
            self.obs_matrix[PIGMENT, curr_x, curr_y] = 1.0
            self.obs_matrix[NO_PIGMENT, curr_x, curr_y] = 0.0
            
    def plot(self, t):
        # https://stackoverflow.com/questions/33282368/plotting-a-2d-heatmap-with-matplotlib
        # y, x = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))

        img = np.zeros((GRID[0], GRID[1]))
        img = img + self.obs_matrix[PIGMENT, :, :]
        fig, ax = plt.subplots()
        ax.imshow(img.T,cmap='gray')
        ax.imshow(self.visit_matrix.T,alpha=0.3)
        ax.scatter([self.x_pos], [self.y_pos], color="red")
        ax.set_title(f"Step: {t}")
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()
        return img

class MDP(object):
    def __init__(self, A, A_2, B, B_2, C):
        self.A = A
        self.A_2 = A_2
        self.B = B
        self.B_2 = B_2
        self.C = C        
        self.p0 = np.exp(-16)

        self.num_states = self.A.shape[1]
        self.num_obs = self.A.shape[0]
        self.num_actions = self.B.shape[0]
        self.num_motifs = self.B_2.shape[0]

        self.prev_motif = 0

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

    def set_A(self, A, A_2):
        self.A = A + self.p0
        self.A = self.normdist(self.A)
        self.lnA = np.log(self.A)

        self.A_2 = A_2 + self.p0
        self.A_2 = self.normdist(self.A_2)
        self.lnA_2 = np.log(self.A_2)

    def reset(self, obs):
        self.curr_obs = obs
        likelihood = self.lnA[obs, :]
        likelihood = likelihood[:, np.newaxis]
        self.sQ = self.softmax(likelihood)
        self.prev_action = self.random_action()

    def step(self, obs, motif_obs,sample=True):        
        # state inference
        likelihood = self.lnA[obs, :]
        likelihood = likelihood[:, np.newaxis]
        prior = np.dot(self.B[self.prev_action], self.sQ)
        prior = np.log(prior)
        # ignore likelihood for now
        # self.sQ = self.softmax(likelihood + prior)
        self.sQ = self.softmax(prior)

        self.likelihood_2 = np.dot(self.lnA_2.T, self.sQ)
        self.prior_2 = np.dot(self.B_2, self.sQ_2)
        self.prior_2 = np.log(self.prior_2 + self.p0)
        # ignore likelihood for now
        self.sQ_2 = self.softmax(self.likelihood_2 + self.prior_2)

        # action inference
        neg_efe = np.zeros([self.num_actions, 1])
        for a in range(self.num_actions):
            fs = np.dot(self.B[a], self.sQ)
            fo = np.dot(self.A, fs)
            fo = self.normdist(fo + self.p0)
            utility = (np.sum(fo * np.log(fo / self.C), axis=0))
            utility = utility[0]
            neg_efe[a] -= utility

            # hack in stay penalty 
            if a is 4:
                stay_penalty = 1000
                neg_efe[a] -= stay_penalty

        # action selection
        self.uQ = self.softmax(neg_efe)
        #print("uQ: ", neg_efe)
        
        if sample == True:
            hu = np.argmax(np.random.multinomial(1,self.uQ))
            #print("hu: ", hu)
            action = hu
        else: 
            #take max
            hu = max(self.uQ)
            options = np.where(self.uQ == hu)[0]
            action = int(np.random.choice(options))
        self.prev_action = action
        #update B_2 matrix statistics
        # dot product self.sq_2 and self.prev_sq_2
        # weight decay , learning rate ?
        self.B_a_2[motif_obs, self.prev_motif] += 0.1
        self.B_2 = self.normdist(np.copy(self.B_a_2))
        self.prev_motif = motif_obs
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
    pprint.pprint(cf)

    A_2 = np.zeros((NUM_STATES, NUM_MOTIFS)) 
    A = np.zeros((NUM_OBSERVATIONS, NUM_STATES))
    #B_2 = np.ones((NUM_MOTIFS, NUM_MOTIFS))
    B_2 = np.eye(NUM_MOTIFS)
    B = np.zeros((NUM_ACTIONS, NUM_STATES, NUM_STATES))
    for a in range(NUM_ACTIONS):
        B[a, a, :] = 1.0

    C = np.zeros((NUM_OBSERVATIONS, 1))
    C[PIGMENT] = 0.9
    C[NO_PIGMENT] = 0.1

    mdp = MDP(A, A_2, B, B_2, C)
    env = Env()

    env.create_motif_env(cf.stim_type)
    # env.create_random_env(num_pixels=GRID_SIZE // 4)
    # init_x = 1
    # init_y = 1
    init_x = GRID[0] // 2
    init_y = GRID[1] // 2

    obs, motif_obs = env.reset(init_x=init_x, init_y=init_y)

    A, A_2 = env.get_A()
    mdp.set_A(A, A_2)
    mdp.reset(obs)
    
    imgs = []
    nb_vert = 0
    nb_horiz = 0
    for t in range(cf.num_steps):
        A, A_2 = env.get_A()
        mdp.set_A(A, A_2)       
        action = mdp.step(obs,motif_obs)
        #print("action: ", action)
        obs, motif_obs = env.step(action)
        img = env.plot(t)
        imgs.append(img)
        #print("motif obs: ",motif_obs)
        #print("sq2: ", mdp.sQ_2)
        #print("B2", mdp.B_2)
        print("pos: ", (env.x_pos, env.y_pos))

        if action == 0 or action == 1 or action == 2 or action == 6 or action == 7 or action == 8:
            nb_vert += 1
        if action == 0 or action == 3 or action == 6 or action == 2 or action == 6 or action == 8:
            nb_horiz += 1
        #print("sq2: ", mdp.sQ_2.shape)
        #print("prior 2: ", mdp.prior_2.shape)
        #print("A2 ", mdp.A_2.shape)
        #print("B2: ",mdp.B_2.shape)
        #print(" likelihood_2 : " ,mdp.likelihood_2.shape)
        #if t > 20:
        #    break
        # focal_area = A[PIGMENT, :].reshape(FOCAL_AREA)
        # pprint.pprint(focal_area)
    
    vertical_index = (GRID[1] * nb_vert - GRID[0] * nb_horiz)/  (GRID[1] * nb_vert + GRID[1] * nb_vert)
    print("Vertical Index: ", vertical_index)

    save_gif(imgs, cf.gif_path)
    return vertical_index, mdp.B_2


cf = AttrDict()
cf.num_steps = 1000
vertical_indices = []
B_matrices = []
for i in range(4):
    cf.stim_type = i
    cf.gif_path = "test_" + str(i) + ".gif"
    vi, B = main(cf)
    np.save("B_matrix_"+str(i)+".npy",B)
    vertical_indices.append(vi)
    B_matrices.append(B)

print("Vertical indicides: ", vertical_indices)
print("B Matrices: ", B_matrices)
vertical_indices = np.array(vertical_indices)
np.save("vertical_indices.npy", vertical_indices)
