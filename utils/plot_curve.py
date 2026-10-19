import numpy as np
import matplotlib.pyplot as plt
import math

# 设置参数
min_w = 0
max_w = 1
curvature = 5  # 快速下降曲线的曲率
t = np.linspace(0, 1, 50)  # 归一化时间
t_normalized = t

print(t_normalized)

# 定义不同的曲线
def linear_curve(t_normalized):
    return min_w + (max_w - min_w) * t_normalized

def constant_curve(t_normalized):
    return (0.5 if t_normalized[0] != 0 else 0) * np.ones_like(t_normalized)

def cosine_curve(t_normalized):
    return min_w + (max_w - min_w) * (1 - np.cos(t_normalized * np.pi / 2))

def cosine_v2_curve(t_normalized):
    return min_w + (max_w - min_w) * (1 - np.cos(t_normalized * np.pi / 2)**2)

def fast_drop_curve(t_normalized):
    return min_w + (max_w - min_w) * np.exp(-curvature * t_normalized)

def betas_curve(t):
    # 示例贝塔曲线数据
    betas = np.linspace(min_w, max_w, len(t))
    return betas

# 绘制曲线
plt.figure(figsize=(10, 6))

# 线性曲线
plt.plot(t, linear_curve(t_normalized), label='Linear Curve', color='yellow')

# 常量曲线
# plt.plot(t, constant_curve(t_normalized), label='Constant Curve', color='orange')

# 余弦曲线
plt.plot(t, cosine_curve(t_normalized), label='Cosine Curve', color='green')

plt.plot(t, cosine_v2_curve(t_normalized), label='Cosine v2 Curve', color='blue')

# 快速下降曲线
plt.plot(t, fast_drop_curve(t_normalized), label='Fast Drop Curve', color='red')

# 贝塔曲线
# plt.plot(t, betas_curve(t), label='Betas Curve', color='purple')

# 添加图例和标签
plt.title('Different Types of Curves')
plt.xlabel('Normalized Time (t)')
plt.ylabel('Weight')
plt.legend()
plt.grid()
# plt.ylim(-0.1, 1.1)  # 设置y轴范围以更好地显示曲线

# 保存图像
plt.savefig('curves_plot.png', dpi=300, bbox_inches='tight')  # 保存为PNG文件

# 显示图像
plt.show()
