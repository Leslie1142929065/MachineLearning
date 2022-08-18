import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import numpy as np #为了生成后面的随机数
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False #显示负号
plt.rcParams['axes.facecolor'] = '00000' #背景板颜色
# 创建画布
fig = plt.figure()
# # 创建3D坐标系
axes3d = Axes3D(fig)
zs = range(6) #X变量类数
left = np.arange(0, 10)
height = np.array([])
for i in range(len(zs)):
    z = zs[i]
    np.random.seed(i)
    height = np.random.randint(0, 50, size=10)
    axes3d.bar(left, height, zs=z, zdir='x',
               color=['red', 'green', 'purple', 'yellow', 'blue', 'black', 'gray', 'orange', 'pink', 'cyan'])
plt.xticks(zs, ['Jan', 'Feb', 'Mar', 'Apr', 'May','Jun'])
plt.yticks(left, ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'])
axes3d.set_zlabel('产量', fontsize=10)
axes3d.set_title('《三维条形图》', y=1.02, fontsize=10, color='00000')
plt.xlabel('月份')
plt.ylabel('型号')
plt.show()
