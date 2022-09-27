from contextlib import closing
from io import StringIO
from os import path
from typing import Optional

import numpy as np

from gym import Env, spaces
from gym.error import DependencyNotInstalled
import netCDF4 as nc


UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3
UPRIGHT = 4
UPLEFT = 5
DOWNRIGHT = 6
DOWNLEFT = 7
def degree2index(deg:float, flag:str):
    if flag == 'N':
        index = deg * 60 + 90 * 60
    if flag == 'S':
        index = 90 * 60 - deg * 60
    if flag == 'E':
        index = deg * 60 + 180 * 60
    if flag == 'W':
        index = 180 * 60 - deg * 60
    return int(index)

def categorical_sample(prob_n, np_random: np.random.Generator):
    """Sample from categorical distribution where each row specifies class probabilities."""
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    return np.argmax(csprob_n > np_random.random())

class shipRouteEnv(Env):
    """
    This is a simple implementation of the Gridworld Cliff
    reinforcement learning task.
    Adapted from Example 6.6 (page 106) from [Reinforcement Learning: An Introduction
    by Sutton and Barto](http://incompleteideas.net/book/bookdraft2018jan1.pdf).
    With inspiration from:
    [https://github.com/dennybritz/reinforcement-learning/blob/master/lib/envs/cliff_walking.py]
    (https://github.com/dennybritz/reinforcement-learning/blob/master/lib/envs/cliff_walking.py)
    ### Description
    The board is a 4x12 matrix, with (using NumPy matrix indexing):
    - [3, 0] as the start at bottom-left
    - [3, 11] as the goal at bottom-right
    - [3, 1..10] as the cliff at bottom-center
    If the agent steps on the cliff, it returns to the start.
    An episode terminates when the agent reaches the goal.
    ### Actions
    There are 4 discrete deterministic actions:
    - 0: move up
    - 1: move right
    - 2: move down
    - 3: move left
    ### Observations
    There are 3x12 + 1 possible states. In fact, the agent cannot be at the cliff, nor at the goal
    (as this results in the end of the episode).
    It remains all the positions of the first 3 rows plus the bottom-left cell.
    The observation is simply the current position encoded as [flattened index](https://numpy.org/doc/stable/reference/generated/numpy.unravel_index.html).
    ### Reward
    Each time step incurs -1 reward, and stepping into the cliff incurs -100 reward.
    ### Arguments
    ```
    gym.make('CliffWalking-v0')
    ```
    ### Version History
    - v0: Initial version release
    """

    metadata = {
        "render_modes": ["human", "rgb_array", "ansi"],
        "render_fps": 4,
    }

    def __init__(self, render_mode: Optional[str] = None):
        # 初始化真实世界中的经纬度 之后的代码会自动将这个经纬度转换为nc数据中的索引
        latstart = 37
        latend = 37.5
        lonstart = 122.5
        lonend = 123
        # 通过init_position.py鼠标手动调整，找到的起止点x y索引坐标
        self.xStartIndex = 2
        self.yStartIndex = 23
        self.xEndIndex = 3
        self.yEndIndex = 2
        # 船舶吃水要求
        self.shipDraught = 5
        # 转换为nc数据中的索引
        latstartIndex = degree2index(latstart, 'N')
        latendIndex = degree2index(latend, 'N')
        lonstartIndex = degree2index(lonstart, 'E')
        lonendIndex = degree2index(lonend, 'E')
        # 计算得到网格形状 1° = 60' nc数据的精度为1' 所以要乘以60
        self.shape = (int((latend - latstart) * 60), int((lonend - lonstart) * 60))
        # np.ravel_multi_index 当第一个参数和第二个参数的shape一致时，起到的作用是，得到(self.yStartIndex, self.xStartIndex)在self.shape这个网格中，从左上角开始按行逐个计数，返回(self.yStartIndex, self.xStartIndex)在新的一维数组中的索引
        # 或者说 np.ravel_multi_index 将self.shape这个网格按行首尾拼接为一维数组，返回原位置为(self.yStartIndex, self.xStartIndex)的数，在新的一维数组中的索引
        # np.ravel_multi_index((self.yStartIndex, self.xStartIndex), self.shape) 等价于 self.yStartIndex * self.shape[0] + self.xStartIndex
        self.start_state_index = np.ravel_multi_index((self.yStartIndex, self.xStartIndex), self.shape)
        # 观察空间
        self.nS = np.prod(self.shape)
        # 动作空间
        self.nA = 4

        # Cliff Location
        self._cliff = np.zeros(self.shape, dtype=bool)

        data = nc.Dataset("ETOPO1_Bed_c_gmt4.grd", "r+")


        self.lon = data.variables['x'][lonstartIndex:lonendIndex]
        self.lat = data.variables['y'][latstartIndex:latendIndex]
        self.dep = data.variables['z'][latstartIndex:latendIndex, lonstartIndex:lonendIndex]
        self.max_x = len(self.lon)
        self.max_y = len(self.lat)
        for i, lon in enumerate(self.lon):
            for j, lat in enumerate(self.lat):
                if self.dep[j, i] > -self.shipDraught:
                    self._cliff[self.max_y - j - 1,i] = True

        # Calculate transition probabilities and rewards
        self.P = {}
        for s in range(self.nS):
            position = np.unravel_index(s, self.shape)
            self.P[s] = {a: [] for a in range(self.nA)}
            self.P[s][UP] = self._calculate_transition_prob(position, [-1, 0])
            self.P[s][RIGHT] = self._calculate_transition_prob(position, [0, 1])
            self.P[s][DOWN] = self._calculate_transition_prob(position, [1, 0])
            self.P[s][LEFT] = self._calculate_transition_prob(position, [0, -1])
            # 计划将动作空间由4个扩展为8个
            # self.P[s][UPRIGHT] = self._calculate_transition_prob(position, [-1, 1])
            # self.P[s][UPLEFT] = self._calculate_transition_prob(position, [-1, -1])
            # self.P[s][DOWNRIGHT] = self._calculate_transition_prob(position, [1, 1])
            # self.P[s][DOWNLEFT] = self._calculate_transition_prob(position, [1, -1])
            
        # Calculate initial state distribution
        self.initial_state_distrib = np.zeros(self.nS)

        self.initial_state_distrib[self.start_state_index] = 1.0

        self.observation_space = spaces.Discrete(self.nS)
        self.action_space = spaces.Discrete(self.nA)

        self.render_mode = render_mode

        # pygame utils
        self.cell_size = (60, 60)
        self.window_size = (
            self.shape[1] * self.cell_size[1],
            self.shape[0] * self.cell_size[0],
        )
        self.window_surface = None
        self.clock = None
        self.elf_images = None
        self.start_img = None
        self.goal_img = None
        self.cliff_img = None
        self.mountain_bg_img = None
        self.near_cliff_img = None
        self.tree_img = None

    def _limit_coordinates(self, coord: np.ndarray) -> np.ndarray:
        """Prevent the agent from falling out of the grid world."""
        coord[0] = min(coord[0], self.shape[0] - 1)
        coord[0] = max(coord[0], 0)
        coord[1] = min(coord[1], self.shape[1] - 1)
        coord[1] = max(coord[1], 0)
        return coord

    def _calculate_transition_prob(self, current, delta):
        """Determine the outcome for an action. Transition Prob is always 1.0.
        Args:
            current: Current position on the grid as (row, col)
            delta: Change in position for transition
        Returns:
            Tuple of ``(1.0, new_state, reward, terminated)``
        """
        new_position = np.array(current) + np.array(delta)
        new_position = self._limit_coordinates(new_position).astype(int)
        new_state = np.ravel_multi_index(tuple(new_position), self.shape)
        if self._cliff[tuple(new_position)]:
            return [(1.0, self.start_state_index, -100, False)]

        terminal_state = (self.yEndIndex, self.xEndIndex)
        is_terminated = tuple(new_position) == terminal_state
        return [(1.0, new_state, -1, is_terminated)]

    def step(self, a):
        transitions = self.P[self.s][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, t = transitions[i]
        self.s = s
        self.lastaction = a

        if self.render_mode == "human":
            self.render()
        return (int(s), r, t, False, {"prob": p})

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.s = categorical_sample(self.initial_state_distrib, self.np_random)
        self.lastaction = None

        if self.render_mode == "human":
            self.render()
        return int(self.s), {"prob": 1}

    def render(self):
        if self.render_mode == "ansi":
            return self._render_text()
        else:
            return self._render_gui(self.render_mode)

    def _render_gui(self, mode):
        try:
            import pygame
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[toy_text]`"
            )
        if self.window_surface is None:
            pygame.init()

            if mode == "human":
                pygame.display.init()
                pygame.display.set_caption("CliffWalking")
                self.window_surface = pygame.display.set_mode(self.window_size)
            else:  # rgb_array
                self.window_surface = pygame.Surface(self.window_size)
        if self.clock is None:
            self.clock = pygame.time.Clock()
        if self.elf_images is None:
            hikers = [
                path.join(path.dirname(__file__), "img/elf_up.png"),
                path.join(path.dirname(__file__), "img/elf_right.png"),
                path.join(path.dirname(__file__), "img/elf_down.png"),
                path.join(path.dirname(__file__), "img/elf_left.png"),
            ]
            self.elf_images = [
                pygame.transform.scale(pygame.image.load(f_name), self.cell_size)
                for f_name in hikers
            ]
        if self.start_img is None:
            file_name = path.join(path.dirname(__file__), "img/stool.png")
            self.start_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.goal_img is None:
            file_name = path.join(path.dirname(__file__), "img/cookie.png")
            self.goal_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.mountain_bg_img is None:
            bg_imgs = [
                path.join(path.dirname(__file__), "img/mountain_bg1.png"),
                path.join(path.dirname(__file__), "img/mountain_bg2.png"),
            ]
            self.mountain_bg_img = [
                pygame.transform.scale(pygame.image.load(f_name), self.cell_size)
                for f_name in bg_imgs
            ]
        if self.near_cliff_img is None:
            near_cliff_imgs = [
                path.join(path.dirname(__file__), "img/mountain_near-cliff1.png"),
                path.join(path.dirname(__file__), "img/mountain_near-cliff2.png"),
            ]
            self.near_cliff_img = [
                pygame.transform.scale(pygame.image.load(f_name), self.cell_size)
                for f_name in near_cliff_imgs
            ]
        if self.cliff_img is None:
            file_name = path.join(path.dirname(__file__), "img/mountain_cliff.png")
            self.cliff_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )

        for s in range(self.nS):
            row, col = np.unravel_index(s, self.shape)
            pos = (col * self.cell_size[0], row * self.cell_size[1])
            check_board_mask = row % 2 ^ col % 2
            self.window_surface.blit(self.mountain_bg_img[check_board_mask], pos)

            if self._cliff[row, col]:
                self.window_surface.blit(self.cliff_img, pos)
            if row < self.shape[0] - 1 and self._cliff[row + 1, col]:
                self.window_surface.blit(self.near_cliff_img[check_board_mask], pos)
            if s == self.start_state_index:
                self.window_surface.blit(self.start_img, pos)
            if s == self.nS - 1:
                self.window_surface.blit(self.goal_img, pos)
            if s == self.s:
                elf_pos = (pos[0], pos[1] - 0.1 * self.cell_size[1])
                last_action = self.lastaction if self.lastaction is not None else 2
                self.window_surface.blit(self.elf_images[last_action], elf_pos)

        if mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window_surface)), axes=(1, 0, 2)
            )

    def _render_text(self):
        outfile = StringIO()

        for s in range(self.nS):
            position = np.unravel_index(s, self.shape)
            if self.s == s:
                output = " x "
            # Print terminal state
            elif position == (3, 11):
                output = " T "
            elif self._cliff[position]:
                output = " C "
            else:
                output = " o "

            if position[1] == 0:
                output = output.lstrip()
            if position[1] == self.shape[1] - 1:
                output = output.rstrip()
                output += "\n"

            outfile.write(output)
        outfile.write("\n")

        with closing(outfile):
            return outfile.getvalue()

if __name__ == "__main__":
    shipRouteEnv(Env)