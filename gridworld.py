import gym
import turtle
import numpy as np
import netCDF4 as nc
from typing import Optional
from shiproute import shipRouteEnv
from PIL import Image
import matplotlib.animation as animation

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
        self.unit = 20
        # 读取全球高程网格nc数据
        data = nc.Dataset("ETOPO1_Bed_c_gmt4.grd", "r+")
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
        # 从数据集中取数据
        self.lon = data.variables['x'][lonstartIndex:lonendIndex]
        self.lat = data.variables['y'][latstartIndex:latendIndex]
        self.dep = data.variables['z'][latstartIndex:latendIndex, lonstartIndex:lonendIndex]
        # self.max_x self.max_y 分别为网格的宽度和高度
        self.max_x = len(self.lon)
        self.max_y = len(self.lat)

    # 绘制网格x轴的网格线
    def draw_x_line(self, y, x0, x1, color='gray'):
        assert x1 > x0
        self.t.color(color)
        self.t.setheading(0)
        self.t.up()
        self.t.goto(x0, y)
        self.t.down()
        self.t.forward(x1 - x0)

    # 绘制网格y轴的网格线
    def draw_y_line(self, x, y0, y1, color='gray'):
        assert y1 > y0
        self.t.color(color)
        self.t.setheading(90)
        self.t.up()
        self.t.goto(x, y0)
        self.t.down()
        self.t.forward(y1 - y0)

    # 绘制不可航区域的黑框
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
    
    # 移动智能体
    def move_player(self, x, y):
        self.t.up()
        self.t.setheading(90)
        self.t.fillcolor('red')
        self.t.goto((x + 0.5) * self.unit, (y + 0.5) * self.unit)
    
    # 渲染网格，渲染智能体采取动作的过程
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
                    y = i * self.unit, x0=0, x1=self.max_x * self.unit)
            for i in range(1, self.max_x):
                self.draw_y_line(
                    x = i * self.unit, y0=0, y1=self.max_y * self.unit)

            # 遍历网格，为不可航区域绘制黑框
            # lon为x(列) lat为y(行)
            # self.dep 是二维数组格式，也是网格的形式，数据为[0,0]的点在左上角，先取行再取列
            # self.draw_box 是坐标轴格式，数据为[0,0]的点在左下角，先取列再取行
            for i, lon in enumerate(self.lon):
                for j, lat in enumerate(self.lat):
                    if self.dep[j, i] > -self.shipDraught:
                        self.draw_box(i, j, 'black')
            # 绘制终点的黄色框
            self.draw_box(self.xEndIndex, self.max_y - 1 - self.yEndIndex, 'yellow')
            self.t.shape('turtle')

        # x_pos y_pos 的表达式不要修改 self.s表示从动作空间中随机取的某一个动作
        x_pos = self.s % self.max_x
        y_pos = self.max_y - 1 - int(self.s / self.max_x)
        self.move_player(x_pos, y_pos)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)


if __name__ == "__main__":
    env = shipRouteEnv()  # 0 up, 1 right, 2 down, 3 left
    env = shipRouteWapper(env)
    env.reset()
    epochs = 100
    for step in range(epochs):
        action = np.random.randint(0, 4)
        # print(env.step(action))
        obs, reward, done, _, info = env.step(action)
        print('step {}: action {}, obs {}, reward {}, done {}, info {}'.format(\
                step, action, obs, reward, done, info))
        env.render()