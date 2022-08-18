import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.facecolor'] = 'white'
fig = plt.figure(figsize=(10, 8), facecolor='white')
ax = Axes3D(fig)
delta = 0.125
# 生成代表X轴数据的列表
x = np.arange(-4.0, 4.0, delta)
# 生成代表Y轴数据的列表
y = np.arange(-3.0, 4.0, delta)
# 对x、y数据执行网格化
X, Y = np.meshgrid(x, y)

Z1 = np.exp(-X**2 - Y**2)
Z2 = np.exp(-(X - 1)**2 - (Y - 1)**2)
# 计算Z轴数据（高度数据）
Z = (Z1 - Z2) * 2
# 绘制3D图形
ax.plot_surface(X, Y, Z,
    rstride=1,  # rstride（row）指定行的跨度
    cstride=1,  # cstride(column)指定列的跨度
    cmap=plt.get_cmap('rainbow'))  # 设置颜色映射
plt.xlabel('X轴', fontsize=15)
plt.ylabel('Y轴', fontsize=15)
ax.set_zlabel('Z轴', fontsize=15)
ax.set_title('《曲面图》', y=1.02, fontsize=25, color='gold')
# 设置Z轴范围
ax.set_zlim(-2, 2)
plt.show()
