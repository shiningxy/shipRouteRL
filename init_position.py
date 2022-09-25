import netCDF4 as nc
import matplotlib.pyplot as plt
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
latstart = degree2index(35, 'N')
latend = degree2index(41, 'N')
lonstart = degree2index(117, 'E')
lonend = degree2index(123, 'E')
LON = data.variables['x'][lonstart:lonend]
LAT = data.variables['y'][latstart:latend]
DEP = data.variables['z'][latstart:latend, lonstart:lonend]

# 构造一个新的船舶可航水域地图 高程高于吃水为不可航区域 在newDep中置为1 高程高于吃水为可航区域 在newDep中置为0
newDep = []
# 船舶吃水要求
shipDraught = 0
for i, lat in enumerate(DEP):
    newLons = []
    for j, lon in enumerate(lat):
        if DEP[i, j] > -shipDraught:
            newLons.append(1)
        else:
            newLons.append(0)
    newDep.append(newLons)

# 画图展示 手动选择起止点坐标
# 例 在此图中选择起始点坐标x=310 y=330 终点坐标x=210 y=50
# 换算为经纬度时除以六十
# 则 新的起始点经纬度应为 shipLatStart = 330/60 + 35 shipLonStart = 310/60 + 117
# 则 新的终点经纬度应为 shipLatEnd = 50/60 + 35 shipLonEnd = 210/60 + 117
# 则 新的起始点坐标应为 shipLatStartIndex = 330 + 35 * 60 shipLonStartIndex = 310 + 117 * 60
# 则 新的终点坐标应为 shipLatEndIndex = 50 + 35 * 60 shipLonEnd = 210 + 117 * 60
# 将这个坐标传参到
ax = plt.subplot()
im = ax.imshow(newDep, cmap='Greys', origin='lower')
plt.show()
