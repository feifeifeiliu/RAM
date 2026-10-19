import torch
from tqdm import tqdm
import os
import sys
from os.path import join as pjoin
import numpy as np

# 获取当前工作目录
current_path = os.getcwd()

# 将当前路径添加到 sys.path 中
if current_path not in sys.path:
    sys.path.append(current_path)


from scripts.motion_process import process_file, recover_from_ric
from common.quaternion import *
from utils.paramUtil import *
from data_loaders.humanml.common.skeleton import Skeleton


if __name__ == "__main__":
    # example_id = "000021"
    # Lower legs
    l_idx1, l_idx2 = 5, 8
    # Right/Left foot
    fid_r, fid_l = [8, 11], [7, 10]
    # Face direction, r_hip, l_hip, sdr_r, sdr_l
    face_joint_indx = [2, 1, 17, 16]
    # l_hip, r_hip
    r_hip, l_hip = 2, 1
    joints_num = 22
    # ds_num = 8
    data_dir = './dataset/Motion-Xplusplus-processed/joint'
    save_dir1 = './dataset/Motion-Xplusplus-processed/new_joints/'
    save_dir2 = './dataset/Motion-Xplusplus-processed/new_joint_vecs/'
    
    os.makedirs(save_dir1, exist_ok=True)
    os.makedirs(save_dir2, exist_ok=True)

    n_raw_offsets = torch.from_numpy(t2m_raw_offsets)
    kinematic_chain = t2m_kinematic_chain
    
    example_id = '_Form_Tai_Chi_Demonstration_Master_form2_grasp_the__birds_tail_right_clip1'

    # Get offsets of target skeleton
    example_data = np.load(os.path.join(data_dir, example_id + '.npy'))
    example_data = example_data.reshape(len(example_data), -1, 3)
    example_data = torch.from_numpy(example_data)
    tgt_skel = Skeleton(n_raw_offsets, kinematic_chain, 'cpu')
    # (joints_num, 3)
    tgt_offsets = tgt_skel.get_offsets_joints(example_data[0])
    print(tgt_offsets)
    
    args = {'tgt_offsets': tgt_offsets,
            'n_raw_offsets': n_raw_offsets}

    source_list = os.listdir(data_dir)
    frame_num = 0
    for source_file in tqdm(source_list):
        source_data = np.load(os.path.join(data_dir, source_file))[:, :joints_num]
        # try:
        data, ground_positions, positions, l_velocity = process_file(source_data, 0.002, args)
        rec_ric_data = recover_from_ric(torch.from_numpy(data).unsqueeze(0).float(), joints_num)
        np.save(pjoin(save_dir1, source_file), rec_ric_data.squeeze().numpy())
        np.save(pjoin(save_dir2, source_file), data)
        frame_num += data.shape[0]
        # except Exception as e:
        #     print(source_file)
        #     print(e)
#         print(source_file)
#         break

    print('Total clips: %d, Frames: %d, Duration: %fm' %
          (len(source_list), frame_num, frame_num / 30 / 60))