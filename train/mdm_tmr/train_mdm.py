# This code is based on https://github.com/openai/guided-diffusion
"""
Train a diffusion model on images.
"""

import os

os.environ['WANDB_BASE_URL'] = 'https://api.bandw.top'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ["HF_HUB_URL"] = "https://hf-mirror.com"

import json
import torch.nn as nn

from utils.fixseed import fixseed
from utils.parser_util import train_args
from utils.config import read_yaml_file, save_yaml_file
from utils import dist_util
from train.mdm_tmr.training_loop import TrainLoop
from data_loaders.get_data import get_dataset_loader
from utils.model_util import create_model_and_diffusion
from utils.model_util import get_model_args
from train.train_platforms import WandBPlatform, ClearmlPlatform, TensorboardPlatform, NoPlatform  # required for the eval operation

def main():
    args = train_args()
    config = read_yaml_file('configs/my_setting.yaml')
    fixseed(args.seed)
    train_platform_type = eval(args.train_platform_type)
    train_platform = train_platform_type(args.save_dir, project=args.project_name)
    train_platform.report_args(args, name='Args')

    if args.save_dir is None:
        raise FileNotFoundError('save_dir was not specified.')
    elif os.path.exists(args.save_dir) and not args.overwrite:
        raise FileExistsError('save_dir [{}] already exists.'.format(args.save_dir))
    elif not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    args_path = os.path.join(args.save_dir, 'args.json')
    with open(args_path, 'w') as fw:
        json.dump(vars(args), fw, indent=4, sort_keys=True)
        
    save_yaml_file(config, os.path.join(args.save_dir, 'my_setting.yaml'))

    dist_util.setup_dist(args.device)

    print("creating data loader...")

    data = get_dataset_loader(name=args.dataset, 
                              batch_size=args.batch_size, 
                              num_frames=args.num_frames, 
                              fixed_len=args.pred_len + args.context_len, 
                              pred_len=args.pred_len,
                              device=dist_util.dev(),
                              generate=True,
                              config=config['data_config'])

    print("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, config, data)
    # nn.DataParallel(model, device_ids=args.device_ids)
    # model = model.cuda()
    model.to(dist_util.dev())
    
    print('Generator params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    TrainLoop(args, config['train_config'], train_platform, model, diffusion, data).run_loop()
    train_platform.close()

if __name__ == "__main__":
    main()
