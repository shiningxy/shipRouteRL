import gym
import turtle
import numpy as np
import netCDF4 as nc
from typing import Optional
from shiproute import shipRouteEnv

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

class shipRouteWapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.t = None
        # unit 为 每个格子的大小 可以单独运行gridworld.py 观察窗口大小 调整self.unit的值
        self.unit = 60
        data = nc.Dataset("ETOPO1_Bed_c_gmt4.grd", "r+")
        latstart = 37.4
        latend = 37.5
        lonstart = 121.7
        lonend = 121.9
        self.xStartIndex = 0
        self.yStartIndex = 0
        self.xEndIndex = 3
        self.yEndIndex = 4
        latstartIndex = degree2index(latstart, 'N')
        latendIndex = degree2index(latend, 'N')
        lonstartIndex = degree2index(lonstart, 'E')
        lonendIndex = degree2index(lonend, 'E')
        self.lon = data.variables['x'][lonstartIndex:lonendIndex]
        self.lat = data.variables['y'][latstartIndex:latendIndex]
        self.dep = data.variables['z'][latstartIndex:latendIndex, lonstartIndex:lonendIndex]
        self.max_x = len(self.lon)
        self.max_y = len(self.lat)

    def draw_x_line(self, y, x0, x1, color='gray'):
        assert x1 > x0
        self.t.color(color)
        self.t.setheading(0)
        self.t.up()
        self.t.goto(x0, y)
        self.t.down()
        self.t.forward(x1 - x0)

    def draw_y_line(self, x, y0, y1, color='gray'):
        assert y1 > y0
        self.t.color(color)
        self.t.setheading(90)
        self.t.up()
        self.t.goto(x, y0)
        self.t.down()
        self.t.forward(y1 - y0)

    def draw_box(self, x, y, fillcolor='', line_color='gray'):
        self.t.up()
        self.t.goto(x * self.unit, y * self.unit)
        self.t.color(line_color)
        self.t.fillcolor(fillcolor)
        self.t.setheading(90)
        self.t.down()
        self.t.begin_fill()
        for i in range(4):
            self.t.forward(self.unit)
            self.t.right(90)
        self.t.end_fill()

    def move_player(self, x, y):
        self.t.up()
        self.t.setheading(90)
        self.t.fillcolor('red')
        self.t.goto((x + 0.5) * self.unit, (y + 0.5) * self.unit)

    def render(self):
        if self.t == None:
            self.t = turtle.Turtle()
            self.wn = turtle.Screen()
            self.wn.setup(self.unit * self.max_x + 100,
                          self.unit * self.max_y + 100)
            self.wn.setworldcoordinates(0, 0, self.unit * self.max_x,
                                        self.unit * self.max_y)
            self.t.shape('circle')
            self.t.width(2)
            self.t.speed(0)
            self.t.color('gray')
            for _ in range(2):
                self.t.forward(self.max_x * self.unit)
                self.t.left(90)
                self.t.forward(self.max_y * self.unit)
                self.t.left(90)
            for i in range(1, self.max_y):
                self.draw_x_line(
                    y=i * self.unit, x0=0, x1=self.max_x * self.unit)
            for i in range(1, self.max_x):
                self.draw_y_line(
                    x=i * self.unit, y0=0, y1=self.max_y * self.unit)

            for i, lon in enumerate(self.lon):
                for j, lat in enumerate(self.lat):
                    if self.dep[j, i] > 0:
                        self.draw_box(i, j, 'black')
            self.draw_box(self.xEndIndex, self.max_y - 1 - self.yEndIndex, 'yellow')
            self.t.shape('turtle')

        x_pos = self.s % self.max_x
        y_pos = self.max_y - 1 - int(self.s / self.max_x)
        self.move_player(x_pos, y_pos)
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

if __name__ == "__main__":
    env = shipRouteEnv()  # 0 up, 1 right, 2 down, 3 left
    env = shipRouteWapper(env)
    env.reset()
    for step in range(50):
        action = np.random.randint(0, 4)
        # print(env.step(action))
        obs, reward, done, _, info = env.step(action)
        print('step {}: action {}, obs {}, reward {}, done {}, info {}'.format(\
                step, action, obs, reward, done, info))
        env.render()