import sys
import types
from pathlib import Path
import torch 
import numpy as np
from omegaconf import DictConfig

# 获取TMR项目的绝对路径
tmr_path = Path(__file__).parent / "TMR"
tmr_path = tmr_path.resolve()

# 将TMR路径添加到sys.path
if str(tmr_path) not in sys.path:
    sys.path.insert(0, str(tmr_path))

from src.load import load_model_from_cfg
from src.config import read_config

def create_tmr_model(cond_mode):
    """创建TMR模型的函数，不依赖Hydra"""
    device = 'cuda'


    if cond_mode == 'tmr-ours-0606':
        run_dir = '/home/yifei/PycharmProjects/Motion-Projects/motion-diffusion-model/utils/TMR/outputs/tmr_humanml3d_guoh3dfeats_d256_bs64_roberta'
        ckpt_name = 'latest-epoch=69-v1'
    elif cond_mode == 'tmr-origin-0606':
        run_dir = '/home/yifei/PycharmProjects/Motion-Projects/motion-diffusion-model/utils/TMR/outputs/tmr_humanml3d_guoh3dfeats'
        ckpt_name = 'last'
    cfg = read_config(run_dir)
    model = load_model_from_cfg(cfg, ckpt_name, eval_mode=True, device=device)
    
    # 文件路径
    mean_pt_path = "/home/yifei/PycharmProjects/Motion-Projects/TMR/stats/humanml3d/guoh3dfeats/mean.pt"
    std_pt_path = "/home/yifei/PycharmProjects/Motion-Projects/TMR/stats/humanml3d/guoh3dfeats/std.pt"
    mean_npy_path = "/home/yifei/PycharmProjects/Motion-Projects/motion-diffusion-model/dataset/t2m_mean.npy"
    std_npy_path = "/home/yifei/PycharmProjects/Motion-Projects/motion-diffusion-model/dataset/t2m_std.npy"
    
    model.mean_tmr = torch.load(mean_pt_path).to(model.device).to(torch.float32)
    model.std_tmr = torch.load(std_pt_path).to(model.device).to(torch.float32)
    model.mean_mdm = torch.from_numpy(np.load(mean_npy_path)).to(model.device).to(torch.float32)
    model.std_mdm = torch.from_numpy(np.load(std_npy_path)).to(model.device).to(torch.float32)
    
    def transform(self, data):
        new_data = data.clone()
        new_data = new_data * self.std_mdm + self.mean_mdm
        new_data = (new_data - self.mean_tmr) / (self.std_tmr + 1e-12)
        return new_data

    model.transform = types.MethodType(transform, model)
    
    return model

import hydra
# 保留原来的Hydra版本（如果需要命令行调用）
@hydra.main(version_base=None, config_path="/home/yifei/PycharmProjects/Motion-Projects/motion-diffusion-model/utils/TMR/configs/model/", config_name="tmr")
def create_tmr_model_with_hydra(newcfg: DictConfig) -> None:
    return create_tmr_model()  # 直接调用不带Hydra的版本

if __name__ == "__main__":
    # 命令行调用时使用Hydra版本
    create_tmr_model_with_hydra()
