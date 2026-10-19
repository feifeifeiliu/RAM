import torch
import numpy as np

# 第一组文件路径（PyTorch格式）
mean_pt_path = "/home/yifei/PycharmProjects/Motion-Projects/TMR/stats/humanml3d/guoh3dfeats/mean.pt"
std_pt_path = "/home/yifei/PycharmProjects/Motion-Projects/TMR/stats/humanml3d/guoh3dfeats/std.pt"

# 第二组文件路径（NumPy格式）
mean_npy_path = "/home/yifei/PycharmProjects/Motion-Projects/motion-diffusion-model/dataset/t2m_mean.npy"
std_npy_path = "/home/yifei/PycharmProjects/Motion-Projects/motion-diffusion-model/dataset/t2m_std.npy"

# 读取数据
try:
    # 读取PyTorch文件
    mean_pt = torch.load(mean_pt_path)
    std_pt = torch.load(std_pt_path)
    
    # 读取NumPy文件
    mean_npy = np.load(mean_npy_path)
    std_npy = np.load(std_npy_path)
    
    # 确保数据类型一致（转换为NumPy数组进行比较）
    if isinstance(mean_pt, torch.Tensor):
        mean_pt = mean_pt.numpy()
    if isinstance(std_pt, torch.Tensor):
        std_pt = std_pt.numpy()
    
    # 计算差值
    mean_diff = np.abs(mean_pt - mean_npy)
    std_diff = np.abs(std_pt - std_npy)
    
    # 输出结果
    print("均值(mean)差值统计:")
    print(f"最大差值: {np.max(mean_diff):.6f}")
    print(f"最小差值: {np.min(mean_diff):.6f}")
    print(f"平均差值: {np.mean(mean_diff):.6f}")
    print(f"差值标准差: {np.std(mean_diff):.6f}")
    print("\n标准差(std)差值统计:")
    print(f"最大差值: {np.max(std_diff):.6f}")
    print(f"最小差值: {np.min(std_diff):.6f}")
    print(f"平均差值: {np.mean(std_diff):.6f}")
    print(f"差值标准差: {np.std(std_diff):.6f}")
    
    # 保存差值结果（可选）
    np.save("mean_diff.npy", mean_diff)
    np.save("std_diff.npy", std_diff)
    print("\n差值结果已保存为mean_diff.npy和std_diff.npy")
    
except Exception as e:
    print(f"处理过程中出错: {e}")