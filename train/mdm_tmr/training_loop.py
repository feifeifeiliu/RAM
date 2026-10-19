import copy
import functools
import os
import time
from types import SimpleNamespace
import numpy as np

import re
from os.path import join as pjoin
from typing import Optional

import blobfile as bf
import torch

from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim

from diffusion import logger
from utils import dist_util
from diffusion.fp16_util import MixedPrecisionTrainer
from diffusion.resample import LossAwareSampler, UniformSampler
from tqdm import tqdm
from diffusion.resample import create_named_schedule_sampler
from data_loaders.humanml.networks.evaluator_wrapper import EvaluatorMDMWrapper
from eval import eval_humanml, eval_humanact12_uestc
# from sample.generate import main as generate
from data_loaders.get_data import get_dataset_loader
from utils.model_util import load_model_wo_clip
from data_loaders.humanml.scripts.motion_process import get_target_location, sample_goal, get_allowed_joint_options
from utils.sampler_util import ClassifierFreeSampleModel
from data_loaders.humanml.scripts.motion_process import recover_from_ric
# from utils.get_tmr_model import create_tmr_model



# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


def get_param_groups(model, weight_decay_list):
    decoder_params = []
    encoder_params = []
    
    for name, param in model.named_parameters():
        if 'decoder' in name:
            decoder_params.append(param)
        elif 'encoder' in name:
            encoder_params.append(param)
    
    param_groups = [
        {
            'params': decoder_params,
            'weight_decay': weight_decay_list[0],
            'name': 'decoder'
        },
        {
            'params': encoder_params,
            'weight_decay': weight_decay_list[1],
            'name': 'encoders'
        }
    ]
    
    return param_groups

def get_param_groups_lr(model, weight_decay_list):
    """
    为模型的不同部分创建参数组，支持不同的权重衰减和学习率
    
    Args:
        model: 模型
        weight_decay_list: [decoder_weight_decay, encoder_weight_decay]
        lr_list: [decoder_lr, encoder_lr]
    """
    decoder_params = []
    encoder_params = []
    
    for name, param in model.named_parameters():
        if 'decoder' in name:
            decoder_params.append(param)
        elif 'encoder' in name:
            encoder_params.append(param)
    
    param_groups = [
        {
            'params': decoder_params,
            'weight_decay': weight_decay_list[0],
            'lr': 2e-4,  # decoder学习率
            'name': 'decoder'
        },
        {
            'params': encoder_params,
            'weight_decay': weight_decay_list[1],
            'lr': 1e-4,  # encoder学习率
            'name': 'encoder'
        }
    ]
    
    return param_groups


class TrainLoop:
    def __init__(self, args, config, train_platform, model, diffusion, data):
        self.args = args
        self.dataset = args.dataset
        self.train_platform = train_platform
        self.model = model
        self.config = config
            
        self.diffusion = diffusion
        self.data = data
        if args.loader_type == 'parallel':
            batch_size = args.batch_size * 2
        else:
            batch_size = args.batch_size
        self.batch_size = batch_size
        self.microbatch = batch_size  # deprecating this option
        self.lr = args.lr
        self.log_interval = args.log_interval
        self.save_interval = args.save_interval
        self.resume_checkpoint = args.resume_checkpoint
        self.amp = True
        if self.amp:
            self.scaler = GradScaler()
        self.weight_decay = args.weight_decay
        self.lr_anneal_steps = args.lr_anneal_steps

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size # * dist.get_world_size()
        self.num_steps = args.num_steps
        self.num_epochs = self.num_steps // len(self.data) + 1

        self.sync_cuda = torch.cuda.is_available()

        
        if self.args.pretrain:
            state_dict = dist_util.load_state_dict('save/humanml_enc_512_50steps/model000750000.pt', map_location=dist_util.dev())
            # import pdb; pdb.set_trace()
            load_model_wo_clip(self.model.motion_decoder, state_dict, train=True, strict=False)
            # state_dict_t = dist_util.load_state_dict('utils/TMR/outputs/tmr_humanml3d_guoh3dfeats/last_weights/text_encoder.pt', map_location=dist_util.dev())
            # load_model_wo_clip(self.model.text_encoder, state_dict_t)
            # state_dict_m = dist_util.load_state_dict('utils/TMR/outputs/tmr_humanml3d_guoh3dfeats/last_weights/motion_encoder.pt', map_location=dist_util.dev())
            # load_model_wo_clip(self.model.motion_encoder, state_dict_m)
            # if self.model.use_motion_encoder and not self.model.share_weight:
            # state_dict = dist_util.load_state_dict('save/humanml_enc_512_50steps/model000750000.pt', map_location=dist_util.dev())
            # load_model_wo_clip(self.model.motion_encoder, state_dict, train=True)
        
        self.model_avg = copy.deepcopy(self.model) if self.args.use_ema else None

        self._load_and_sync_parameters()

        self.save_dir = args.save_dir
        self.overwrite = args.overwrite

        if self.args.use_ema:
            # self.opt_G = AdamW(
            #     # with amp, we don't need to use the mp_trainer's master_params
            #     self.model.parameters(),
            #     lr=self.lr,
            #     weight_decay=self.weight_decay,
            #     betas=(0.9, self.args.adam_beta2),
            # )
            param_groups = get_param_groups(self.model, self.weight_decay)
            self.opt_G = AdamW(
                param_groups,
                lr=self.lr,
                betas=(0.9, self.args.adam_beta2),
            )
        else:
            self.opt_G = AdamW(
                self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
            
        # self.scheduler = optim.lr_scheduler.MultiStepLR(self.opt_t2m_transformer,
        #                                         milestones=self.opt.milestones,
        #                                         gamma=self.opt.gamma)
            
        self.model_for_eval = self.model_avg if self.args.use_ema else self.model
        if args.gen_guidance_param != 1:
            self.model_for_eval = ClassifierFreeSampleModel(self.model_for_eval.motion_decoder)   # wrapping model with the classifier-free sampler

        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.

        self.device = torch.device("cpu")
        if torch.cuda.is_available() and dist_util.dev() != 'cpu':
            self.device = torch.device(dist_util.dev())

        self.schedule_sampler_type = 'uniform'
        self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_type, diffusion)
        self.eval_wrapper, self.eval_data, self.eval_gt_data = None, None, None
        if args.dataset in ['kit', 'humanml'] and args.eval_during_training:
            mm_num_samples = 0  # mm is super slow hence we won't run it during training
            mm_num_repeats = 0  # mm is super slow hence we won't run it during training
            gen_loader = get_dataset_loader(name=args.dataset, batch_size=args.eval_batch_size, num_frames=None,
                                            split=args.eval_split,
                                            hml_mode='eval',
                                            autoregressive=args.autoregressive,
                                            fixed_len=args.context_len+args.pred_len, pred_len=args.pred_len, device=dist_util.dev())

            self.eval_gt_data = get_dataset_loader(name=args.dataset, batch_size=args.eval_batch_size, num_frames=None,
                                                   split=args.eval_split,
                                                   hml_mode='gt', device=dist_util.dev())
            self.eval_wrapper = EvaluatorMDMWrapper(args.dataset, dist_util.dev())
            self.eval_data = {
                'test': lambda: eval_humanml.get_mdm_loader(self.args,
                    self.model_for_eval, diffusion, args.eval_batch_size,
                    gen_loader, mm_num_samples, mm_num_repeats, gen_loader.dataset.opt.max_motion_length,
                    args.eval_num_samples, scale=args.gen_guidance_param,
                )
            }
        self.use_ddp = False
        self.ddp_model = self.model


    def _load_and_sync_parameters(self):
        resume_checkpoint = self.find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            # we add 1 because self.resume_step has already been done and we don't want to run it again
            # in particular we don't want to run the evaluation and generation again
            self.step += 1  
            
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint) 
            logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
            state_dict = dist_util.load_state_dict(
                resume_checkpoint, map_location=dist_util.dev())

            if 'model_avg' in state_dict:
                print('loading both model and model_avg')
                state_dict, state_dict_avg = state_dict['model'], state_dict[
                    'model_avg']
                load_model_wo_clip(self.model, state_dict)
                load_model_wo_clip(self.model_avg, state_dict_avg)
            else:
                load_model_wo_clip(self.model, state_dict)
                if self.args.use_ema:
                    # in case we load from a legacy checkpoint, just copy the model
                    print('loading model_avg from model')
                    self.model_avg.load_state_dict(self.model.state_dict(), strict=False)

            # self.model.load_state_dict(
            #     dist_util.load_state_dict(
            #         resume_checkpoint, map_location=dist_util.dev()
            #     ), strict=False
            # )

    def _load_optimizer_state(self):
        main_checkpoint = self.find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:09}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )

            if self.amp:
                if 'scaler' not in state_dict:
                    print("scaler state not found ... not loading it.")
                    state_dict_G = state_dict['opt_G']
                else:
                    # load grad scaler state
                    self.scaler.load_state_dict(state_dict['scaler'])
                    # for the rest
                    state_dict_G = state_dict['opt_G']
            state_dict_G = state_dict['opt_G']

            tgt_wd = self.opt_G.param_groups[0]['weight_decay']
            print('target weight decay:', tgt_wd)
            self.opt_G.load_state_dict(state_dict_G)
            # self.opt_D.load_state_dict(state_dict_D)
            print('loaded weight decay (will be replaced):',
                  self.opt_G.param_groups[0]['weight_decay'])
            # preserve the weight decay parameter
            for group in self.opt_G.param_groups:
                group['weight_decay'] = tgt_wd
            # for group in self.opt_D.param_groups:
            #     group['weight_decay'] = tgt_wd
            self.opt_G.param_groups[0]['capturable'] = True
            # self.opt_D.param_groups[0]['capturable'] = True

    def cond_modifiers(self, cond, motion):
        # All modifiers must be in-place
        self.target_cond_modifier(cond, motion)
    
    def target_cond_modifier(self, cond, motion):
        if self.args.multi_target_cond:
            batch_size = motion.shape[0]
            cond['target_joint_names'], cond['is_heading'] = sample_goal(batch_size, motion.device, self.args.target_joint_names)

            cond['target_cond'] = get_target_location(motion, 
                                                      self.data.dataset.mean[None, :, None, None], 
                                                      self.data.dataset.std[None, :, None, None], 
                                                      cond['lengths'], 
                                                      self.data.dataset.t2m_dataset.opt.joints_num, self.model.all_goal_joint_names, cond['target_joint_names'], cond['is_heading']).detach()

    def run_loop(self):
        print('train steps:', self.num_steps)
        for epoch in range(self.num_epochs):
            print(f'Starting epoch {epoch}')
            for motion, cond in tqdm(self.data):
                if not (not self.lr_anneal_steps or self.total_step() < self.lr_anneal_steps):
                    break
                
                self.cond_modifiers(cond['y'], motion) # Modify in-place for efficiency
                motion = motion.to(self.device)
                cond['y'] = {key: val.to(self.device) if torch.is_tensor(val) else val for key, val in cond['y'].items()}

                self.run_step(motion, cond)
                if self.total_step() % self.log_interval == 0:
                    for k,v in logger.get_current().dumpkvs().items():
                        if k == 'loss':
                            print('step[{}]: loss[{:0.5f}]'.format(self.total_step(), v))

                        if k in ['step', 'samples']: # or '_q' in k:
                            continue
                        else:
                            self.train_platform.report_scalar(name=k, value=v, iteration=self.total_step(), group_name='Loss')

                if self.total_step() % self.save_interval == 0:
                    self.save()
                    self.model.eval()
                    if self.args.use_ema:
                        self.model_avg.eval()
                    self.evaluate()
                    self.generate_during_training()
                    self.model.train()
                    if self.args.use_ema:
                        self.model_avg.train()

                    # Run for a finite amount of time in integration tests.
                    if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.total_step() > 0:
                        return
                self.step += 1
            if not (not self.lr_anneal_steps or self.total_step() < self.lr_anneal_steps):
                break
        # Save the last checkpoint if it wasn't already saved.
        if (self.total_step() - 1) % self.save_interval != 0:
            self.save()
            self.evaluate()

    def evaluate(self):
        if not self.args.eval_during_training:
            return
        start_eval = time.time()
        if self.eval_wrapper is not None:
            print('Running evaluation loop: [Should take about 90 min]')
            log_file = os.path.join(self.save_dir, f'eval_humanml_{(self.total_step()):09d}.log')
            diversity_times = 300
            mm_num_times = 0  # mm is super slow hence we won't run it during training
            eval_dict = eval_humanml.evaluation(
                self.eval_wrapper, self.eval_gt_data, self.eval_data, log_file,
                replication_times=self.args.eval_rep_times, diversity_times=diversity_times, mm_num_times=mm_num_times, run_mm=False)
            print(eval_dict)
            for k, v in eval_dict.items():
                if k.startswith('R_precision'):
                    for i in range(len(v)):
                        self.train_platform.report_scalar(name=f'top{i + 1}_' + k, value=v[i],
                                                          iteration=self.total_step(),
                                                          group_name='Eval')
                else:
                    self.train_platform.report_scalar(name=k, value=v, iteration=self.total_step(),
                                                      group_name='Eval')

        elif self.dataset in ['humanact12', 'uestc']:
            eval_args = SimpleNamespace(num_seeds=self.args.eval_rep_times, num_samples=self.args.eval_num_samples,
                                        batch_size=self.args.eval_batch_size, device=self.device, guidance_param = 1,
                                        dataset=self.dataset, unconstrained=self.args.unconstrained,
                                        model_path=os.path.join(self.save_dir, self.ckpt_file_name()))
            eval_dict = eval_humanact12_uestc.evaluate(eval_args, model=self.model, diffusion=self.diffusion, data=self.data.dataset)
            print(f'Evaluation results on {self.dataset}: {sorted(eval_dict["feats"].items())}')
            for k, v in eval_dict["feats"].items():
                if 'unconstrained' not in k:
                    self.train_platform.report_scalar(name=k, value=np.array(v).astype(float).mean(), iteration=self.step, group_name='Eval')
                else:
                    self.train_platform.report_scalar(name=k, value=np.array(v).astype(float).mean(), iteration=self.step, group_name='Eval Unconstrained')

        end_eval = time.time()
        print(f'Evaluation time: {round(end_eval-start_eval)/60}min')


    def run_step(self, batch, cond):
        self.opt_G.zero_grad()
        
        t_enc, _ = self.schedule_sampler.sample(batch.shape[0], dist_util.dev(), self.config['encoder_t_step_upper'])
        t_dec, _ = self.schedule_sampler.sample(batch.shape[0], dist_util.dev())
        # t_enc = t_dec = None
        cond['y']['step'] = self.total_step()
        
        compute_losses = functools.partial(
            self.diffusion.training_losses,
            self.ddp_model,
            batch,  # [bs, ch, image_size, image_size]
            t_enc,
            t_dec,# [bs](int) sampled timesteps
            model_kwargs=cond,
            dataset=self.data.dataset
        )
        
        if self.amp:
            with autocast():
                losses = compute_losses()
            
            self.scaler.scale(losses['loss']).backward()  # scaler实现的反向误差传播
            self.scaler.step(self.opt_G)  # 优化器中的值也需要放缩
            self.scaler.update()  # 更新scaler
        else:
        
            losses = compute_losses()
            losses['loss'].backward()
            self.opt_G.step()
        
        log_loss_dict(
            self.diffusion, t_dec, {k: v for k, v in losses.items()}
        )
       
        self.update_average_model()
        self._anneal_lr()
        self.log_step()
        

    def update_average_model(self):
        # update the average model using exponential moving average
        if self.args.use_ema:
            # master params are FP32
            params = self.model.parameters()
            for param, avg_param in zip(params, self.model_avg.parameters()):
                # avg = avg + (param - avg) * (1 - alpha)
                # avg = avg + param * (1 - alpha) - (avg - alpha * avg)
                # avg = alpha * avg + param * (1 - alpha)
                avg_param.data.mul_(self.args.avg_model_beta).add_(
                    param.data, alpha=1 - self.args.avg_model_beta)
                


    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = self.total_step() / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt_G.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.total_step())
        logger.logkv("samples", (self.total_step() + 1) * self.global_batch)

    def ckpt_file_name(self):
        return f"model{(self.total_step()):09d}.pt"

  
    def generate_during_training(self):
        if not self.args.gen_during_training:
            return
        gen_args = copy.deepcopy(self.args)
        gen_args.model_path = os.path.join(self.save_dir, self.ckpt_file_name())
        gen_args.output_dir = os.path.join(self.save_dir, f'{self.ckpt_file_name()}.samples')
        gen_args.num_samples = self.args.gen_num_samples
        gen_args.num_repetitions = self.args.gen_num_repetitions
        gen_args.guidance_param = self.args.gen_guidance_param
        gen_args.motion_length = 6  # fixed length
        gen_args.input_text = gen_args.text_prompt = gen_args.action_file = gen_args.action_name = gen_args.dynamic_text_path = ''
        if gen_args.multi_target_cond:
            gen_args.sampling_mode = 'goal'
            gen_args.target_joint_source = 'data'
        # all_sample_save_path = generate(gen_args)
        # self.train_platform.report_media(title='Motion', series='Predicted Motion', iteration=self.total_step(),
        #                                  local_path=all_sample_save_path)        

    
    def find_resume_checkpoint(self) -> Optional[str]:
        '''look for all file in save directory in the pattent of model{number}.pt
            and return the one with the highest step number.

        TODO: Implement this function (alredy existing in MDM), so that find model will call it in case a ckpt exist.
        TODO: Change call for find_resume_checkpoint and send save_dir as arg.
        TODO: This means ignoring the flag of resume_checkpoint in case some other ckpts exists in that dir!
        '''

        matches = {file: re.match(r'model(\d+).pt$', file) for file in os.listdir(self.args.save_dir)}
        models = {int(match.group(1)): file for file, match in matches.items() if match}

        return pjoin(self.args.save_dir, models[max(models)]) if models else None
    
    def total_step(self):
        return self.step + self.resume_step
    
    def save(self):
        def save_checkpoint():
            def del_clip(state_dict):
                # Do not save CLIP weights
                clip_weights = [
                    e for e in state_dict.keys() if e.startswith('clip_model.')
                ]
                for e in clip_weights:
                    del state_dict[e]


            state_dict = self.model.state_dict()
            del_clip(state_dict)

            if self.args.use_ema:
                # save both the model and the average model
                state_dict_avg = self.model_avg.state_dict()
                del_clip(state_dict_avg)
                state_dict = {'model': state_dict, 'model_avg': state_dict_avg,}

            logger.log(f"saving model...")
            filename = self.ckpt_file_name()
            with bf.BlobFile(bf.join(self.save_dir, filename), "wb") as f:
                torch.save(state_dict, f)

        save_checkpoint()

        with bf.BlobFile(
            bf.join(self.save_dir, f"opt{(self.total_step()):09d}.pt"),
            "wb",
        ) as f:
            opt_G_state = self.opt_G.state_dict()
            # opt_D_state = self.opt_D.state_dict()
            if self.amp:
                # with fp16 we also save the state dict
                opt_state = {
                    'opt_G': opt_G_state,
                    # 'opt_D': opt_D_state,
                    'scaler': self.scaler.state_dict(),
                }
            else:
                opt_state = {
                    'opt_G': opt_G_state,
                    # 'opt_D': opt_D_state,
                }

            torch.save(opt_state, f)
            
            
    def update_lr_warm_up(self, nb_iter, warm_up_iter, lr):

        current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
        for param_group in self.model.param_groups:
            param_group["lr"] = current_lr

        return current_lr


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()



def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        # if key in ['rot_mse_p']:
        #     for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
        #         quartile = int(4 * sub_t / diffusion.num_timesteps)
        #         logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
    

def sample_frames(x, lengths, n=1):
    batch_size, feature_dim, _, max_frames = x.shape
    # 筛选出有效长度大于等于 10 的样本索引
    valid_indices = torch.where(lengths >= 10)[0]
    valid_lengths = lengths[valid_indices]

    all_start_positions = []
    for length in valid_lengths:
        start_positions = torch.randint(0, length - 10 + 1, (n,))
        all_start_positions.append(start_positions)
    all_start_positions = torch.cat(all_start_positions)

    # 扩展有效样本索引以匹配每个采样
    expanded_valid_indices = valid_indices.repeat_interleave(n)

    # 生成切片索引
    indices = all_start_positions.unsqueeze(1) + torch.arange(10).unsqueeze(0)

    # 采样帧
    sampled_frames = x[expanded_valid_indices.unsqueeze(1).unsqueeze(2).unsqueeze(3),
                       torch.arange(feature_dim).unsqueeze(0).unsqueeze(2).unsqueeze(3),
                       0,
                       indices.unsqueeze(1).unsqueeze(2)]

    # 调整形状为 [batch_size * n, feature_dim, 1, 10]
    # sampled_frames = sampled_frames.permute(0, 2, 3, 1)

    return sampled_frames




def log_norm(model):
    grad_norm = 0.0
    param_norm = 0.0
    for p in list(model.parameters()):
        with torch.no_grad():
            param_norm += torch.norm(p, p=2, dtype=torch.float32).item() ** 2
            if p.grad is not None:
                grad_norm += torch.norm(p.grad, p=2, dtype=torch.float32).item() ** 2
    logger.logkv_mean("d_grad_norm", grad_norm)
    logger.logkv_mean("d_param_norm", param_norm)
    
def get_grad_norm(model):
    total_gradient = 0
    count = 0
    
    for param in model.parameters():
        if param.grad is not None:
            # 累加梯度的绝对值之和（也可直接累加梯度值）
            total_gradient += param.grad.abs().sum().item()
            count += param.numel()  # 参数总数
    
    if count == 0:
        return 0
    
    # 返回所有参数梯度的平均绝对值
    return total_gradient / count
