# shipRouteRL

## Download

下载[ETOPO1_Bed_c_gmt4.grd.gz](https://ngdc.noaa.gov/mgg/global/relief/ETOPO1/data/bedrock/cell_registered/netcdf/ETOPO1_Bed_c_gmt4.grd.gz)，存放至根目录

## Install
```
conda create -n shiprl python=3.7
conda activate shiprl
pip install parl -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install visualdl -i https://mirror.baidu.com/pypi/simple
pip install gym==0.26.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install netCDF4 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install pygame -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install pandas -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install seaborn -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## TODO

gridworld.py & shiproute.py中的这九个参数，需要预先使用init_position.py进行选择并计算确定，之后手动修改

先确定网格范围，再选取起止点索引坐标

```
latstart = 37
latend = 37.5
lonstart = 122.5
lonend = 123
self.xStartIndex = 0
self.yStartIndex = 21
self.xEndIndex = 9
self.yEndIndex = 4
self.shipDraught = 5
```

## Structure

main.py -> 主程序入口，完成训练和测试

init_position -> 用于确定网格范围和起止点索引坐标

gridworld.py -> 继承gym.Wrapper类，构建网格，可单独运行查看渲染窗口的大小是否合适

shiproute.py -> 继承gym.Env类，构建环境，定义动作空间

agent.py -> 定义Qlearning智能体

utils.py -> 存储训练过程和测试结果，将reward绘制出来

VisualDL可视化分析工具使用介绍.ipynb -> visualdl的训练过程展示，类似tensorboard 效果更美观