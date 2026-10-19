import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegFileWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.axes3d as p3
# import cv2
from textwrap import wrap


def list_cut_average(ll, intervals):
    if intervals == 1:
        return ll

    bins = math.ceil(len(ll) * 1.0 / intervals)
    ll_new = []
    for i in range(bins):
        l_low = intervals * i
        l_high = l_low + intervals
        l_high = l_high if l_high < len(ll) else len(ll)
        ll_new.append(np.mean(ll[l_low:l_high]))
    return ll_new


def plot_3d_motion(save_path, kinematic_tree, joints, title, dataset, figsize=(3, 3), fps=120, radius=3,
                   vis_mode='default', gt_frames=[]):
    matplotlib.use('Agg')

    # 处理长标题
    title_lines = wrap(title, 30)  # 每行最多30个字符
    title = '\n'.join(title_lines)
    
    def init():
        # 调整画布布局
        fig.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.9)
        
        # 调整3D视图位置
        ax.set_position([0.1, 0.1, 0.8, 0.7])  # [left, bottom, width, height]
        
        # 调整坐标轴范围
        ax.set_xlim3d([-radius/2, radius/2])
        ax.set_ylim3d([-radius/2, radius/2])  # 动作显示区域下移
        ax.set_zlim3d([-radius/3, radius*2/3])
        
        # 标题设置
        fig.suptitle(title, fontsize=8, y=0.95)  # 标题位置上移
        ax.grid(b=False)
        return ax

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz]
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)

    data = joints.copy().reshape(len(joints), -1, 3)

    if dataset == 'kit':
        data *= 0.003
    elif dataset == 'humanml':
        data *= 1.3
    elif dataset in ['humanact12', 'uestc']:
        data *= -1.5

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    colors_blue = ["#4D84AA", "#5B9965", "#61CEB9", "#34C1E2", "#80B79A"]
    colors_orange = ["#DD5A37", "#D69E00", "#B75A39", "#FF6D00", "#DDB50E"]
    colors = colors_orange

    frame_number = data.shape[0]
    height_offset = MINS[1]
    # data[:, :, 1] -= height_offset
    # data[:, :, 1] -= radius/2
    trajec = data[:, 0, [0, 2]]
    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]

    def update(index):
        try:
            ax.clear()
            init()
            plot_xzPlane(MINS[0] - trajec[index, 0], MAXS[0] - trajec[index, 0], 0,  # y轴起点下移
                         MINS[2] - trajec[index, 1], MAXS[2] - trajec[index, 1])

            used_colors = colors_blue if index in gt_frames else colors
            for i, (chain, color) in enumerate(zip(kinematic_tree, used_colors)):
                linewidth = 4.0 if i < 5 else 2.0
                ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2], 
                          linewidth=linewidth, color=color)

            ax.view_init(elev=120, azim=-90)
            ax.dist = 7.5
            plt.axis('off')
        except Exception as e:
            print(f"Error in frame {index}: {str(e)}")

    ani = FuncAnimation(fig, update, frames=frame_number, interval=1000 / fps, repeat=False)
    ani.save(save_path, fps=fps)
    plt.close()