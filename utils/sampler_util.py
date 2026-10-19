import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from utils.misc import wrapped_getattr
import joblib

# A wrapper model for Classifier-free guidance **SAMPLING** only
# https://arxiv.org/abs/2207.12598
class ClassifierFreeSampleModel(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model  # model is the actual model to run

        assert self.model.cond_mask_prob > 0, 'Cannot run a guided diffusion on a model that has not been trained with no conditions'

        # pointers to inner model
        self.translation = self.model.translation
        self.njoints = self.model.njoints
        self.nfeats = self.model.nfeats
        self.data_rep = self.model.data_rep
        self.original_format = True
        self.dropout_guidance = False

        # REG trajectory logging (OFF by default — zero cost when disabled).
        # When enabled, every forward() call appends one dict to self.reg_log
        # with per-sample norms and encoded latents used by Fig. R_REG.
        self.enable_reg_log = False
        self.reg_log = []

    def reset_reg_log(self):
        self.reg_log = []

    def _frob_norm_per_sample(self, t):
        # t: [B, nfeats, 1, nframes] -> [B]
        return t.detach().flatten(1).norm(dim=1).cpu()

    def _encode_motion(self, motion_bnf1t, m_mask_bt):
        # motion_bnf1t: [B, nfeats, 1, nframes]; m_mask_bt: [B, nframes]
        motion_bnt_f = motion_bnf1t.detach().squeeze(2).permute(0, 2, 1)
        enc_in = {'x': motion_bnt_f, 'm_mask': m_mask_bt}
        return self.model.encode(enc_in, sample_mean=True).detach().cpu()

    def forward(self, x, timesteps, y=None):
        cfg_scale = y['cfg_scale'].view(-1, 1, 1, 1)
        reg_scale = y['reg_scale'].view(-1, 1, 1, 1)
        
        threshold = 0.
        
        # y_uncond = deepcopy(y)
        # y_uncond['uncond'] = True
        # out = self.model(x, timesteps, y)
        # out_uncond = self.model(x, timesteps, y_uncond)
        
        # predict = out if self.original_format else out_uncond
        # predict = predict + (cfg_scale * (out - out_uncond))
        
        # if 'recon_latent' in y.keys():
        #     y_motion = deepcopy(y)
        #     y_motion['latent'] = y_motion['recon_latent']
        #     out_motion = self.model(x, timesteps, y_motion)
            
        #     predict = predict + reg_scale * (out - out_motion)
        
        y_uncond = deepcopy(y)
        y_uncond['uncond'] = True
        out = self.model(x, timesteps, y)
        out_uncond = None
        out_motion = None

        if cfg_scale[0,0,0,0].item()>0:
            out_uncond = self.model(x, timesteps, y_uncond)

            diff_uncond = out - out_uncond
            mask_uncond = (diff_uncond.abs() > threshold).float()
            diff_uncond = diff_uncond * mask_uncond

            predict = out if self.original_format else out_uncond
            predict = predict + (cfg_scale * diff_uncond)
        else:
            predict = out


        # if reg_scale[0] > 0:
        #     motion_x_dict = {'x': out.squeeze().permute(0,2,1), 'm_mask': y['mask'].squeeze()}
        #     y['recon_latent'] = self.model.encode(motion_x_dict, sample_mean=True)

        if 'recon_latent' in y.keys() and reg_scale[0,0,0,0].item()>0:
            y_motion = deepcopy(y)
            y_motion['latent'] = y_motion['recon_latent']
            out_motion = self.model(x, timesteps, y_motion)

            diff_motion = out - out_motion
            mask_motion = (diff_motion.abs() > threshold).float()
            diff_motion = diff_motion * mask_motion

            predict = predict + reg_scale * diff_motion

        if 'pre_pred' in y.keys():
            predict = predict + reg_scale * (out - y['pre_pred'])

        if self.enable_reg_log:
            m_mask_bt = y['mask'].squeeze(1).squeeze(1)
            entry = {
                't': int(timesteps[0].item()),
                'cfg_scale': float(cfg_scale[0, 0, 0, 0].item()),
                'reg_scale': float(reg_scale[0, 0, 0, 0].item()),
                'diff_uncond_norm': (
                    self._frob_norm_per_sample(out - out_uncond)
                    if out_uncond is not None else None
                ),
                'diff_motion_norm': (
                    self._frob_norm_per_sample(out - out_motion)
                    if out_motion is not None else None
                ),
                'z_text':   self._encode_motion(out,     m_mask_bt),
                'z_final':  self._encode_motion(predict, m_mask_bt),
                'z_anchor': (
                    y['recon_latent'].detach().cpu()
                    if 'recon_latent' in y else None
                ),
                'z_uncond': (
                    self._encode_motion(out_uncond, m_mask_bt)
                    if out_uncond is not None else None
                ),
                'z_motion': (
                    self._encode_motion(out_motion, m_mask_bt)
                    if out_motion is not None else None
                ),
                # Motion-space (pre-encoder) predict_i for Panel A error-pattern
                # analysis. fp16 to keep [S, B, nfeats, T] dump under 1 GB.
                'predict_raw': predict.detach().squeeze(2).cpu().half(),
                # Frame validity mask, [B, T] bool. Same across steps but cheap.
                'm_mask_bt': m_mask_bt.detach().cpu().bool(),
            }
            self.reg_log.append(entry)

        return predict

    def __getattr__(self, name, default=None):
        # this method is reached only if name is not in self.__dict__.
        return wrapped_getattr(self, name, default=None)
    


class AutoRegressiveSampler():
    def __init__(self, args, sample_fn, required_frames=196):
        self.sample_fn = sample_fn
        self.args = args
        self.required_frames = required_frames
    
    def sample(self, model, shape, **kargs):
        bs = shape[0]
        n_iterations = (self.required_frames // self.args.pred_len) + int(self.required_frames % self.args.pred_len > 0)
        samples_buf = []
        cur_prefix = deepcopy(kargs['model_kwargs']['y']['prefix'])  # init with data
        dynamic_text_mode = type(kargs['model_kwargs']['y']['text'][0]) == list  # Text changes on the fly - prompt per prediction is provided as a list (instead of a single prompt)
        if self.args.autoregressive_include_prefix:
            samples_buf.append(cur_prefix)
        autoregressive_shape = list(deepcopy(shape))
        autoregressive_shape[-1] = self.args.pred_len
        
        # Autoregressive sampling
        for i in range(n_iterations):
            
            # Build the current kargs
            cur_kargs = deepcopy(kargs)
            cur_kargs['model_kwargs']['y']['prefix'] = cur_prefix
            if dynamic_text_mode:
                cur_kargs['model_kwargs']['y']['text'] = [s[i] for s in kargs['model_kwargs']['y']['text']]
                if model.text_encoder_type == 'bert':
                    cur_kargs['model_kwargs']['y']['text_embed'] = (cur_kargs['model_kwargs']['y']['text_embed'][0][:, :, i], cur_kargs['model_kwargs']['y']['text_embed'][1][:, i])
                else:
                    raise NotImplementedError('DiP model only supports BERT text encoder at the moment. If you implement this, please send a PR!')
            
            # Sample the next prediction
            sample = self.sample_fn(model, autoregressive_shape, **cur_kargs)

            # Buffer the sample
            samples_buf.append(sample.clone()[..., -self.args.pred_len:])

            # Update the prefix
            cur_prefix = sample.clone()[..., -self.args.context_len:]

        full_batch = torch.cat(samples_buf, dim=-1)[..., :self.required_frames]  # 200 -> 196
        return full_batch
    

def enable_dropout(model):
    """启用模型中的dropout层"""
    total_num = 0
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            total_num += 1
            if total_num < 5:
                module.train(True)