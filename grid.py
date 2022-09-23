import netCDF4 as nc

data = nc.Dataset("ETOPO1_Bed_c_gmt4.grd", "r+")
lon = data.variables['x']
lat = data.variables['y']
dep = data.variables['z']
# lat -90 ~ 90  1'
# lon -180 ~ 180  1'
# 渤海
print(lon[17960:17990])
print(lat[7620:7650])
print(dep[7620:7650, 17960:17990])
