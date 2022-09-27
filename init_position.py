import matplotlib.pyplot as plt
import numpy as np
import netCDF4 as nc
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
# 读取全球高程数据 1'精度
data = nc.Dataset("ETOPO1_Bed_c_gmt4.grd", "r+")

# 渤海
# z : y, x 
# 北纬 and 东经
# TODO : 输入经纬度范围
latstart = 37
latend = 37.5
lonstart = 122.5
lonend = 123
xStartIndex = 2
yStartIndex = 23
xEndIndex = 3
yEndIndex = 2
shipDraught = 5
latstartIndex = degree2index(latstart, 'N')
latendIndex = degree2index(latend, 'N')
lonstartIndex = degree2index(lonstart, 'E')
lonendIndex = degree2index(lonend, 'E')
LON = data.variables['x'][lonstartIndex:lonendIndex]
LAT = data.variables['y'][latstartIndex:latendIndex]
DEP = data.variables['z'][latstartIndex:latendIndex, lonstartIndex:lonendIndex]

# 船舶吃水要求

shape = (int((latend - latstart) * 60), int((lonend - lonstart) * 60))
latstartIndex = degree2index(latstart, 'N')
latendIndex = degree2index(latend, 'N')
lonstartIndex = degree2index(lonstart, 'E')
lonendIndex = degree2index(lonend, 'E')
lons = data.variables['x'][lonstartIndex:lonendIndex]
lats = data.variables['y'][latstartIndex:latendIndex]
dep = data.variables['z'][latstartIndex:latendIndex, lonstartIndex:lonendIndex]
max_x = len(lons)
max_y = len(lats)
_cliff = np.zeros(shape)
for i, lon in enumerate(lons):
    for j, lat in enumerate(lats):
        if dep[j, i] >= -shipDraught:
            _cliff[max_y - j - 1,i] = True

print(_cliff[yStartIndex, xStartIndex])
print(_cliff[yEndIndex, xEndIndex])
ax = plt.subplot()
im = ax.imshow(_cliff, cmap='Greys', origin='upper')
plt.show()

# 画图展示 手动选择起止点坐标
# 例 在此图中选择起始点坐标x=310 y=330 终点坐标x=210 y=50 网格大小360*360
# self.xStartIndex = 310
# self.yStartIndex = 360 - 1 - 330
# self.xEndIndex = 210
# self.yEndIndex = 360 - 1 - 50
