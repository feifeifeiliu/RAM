import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, List
import copy


from model.mdm_decoder import MDMDecoder
from model.tmr_model.actor import ACTORStyleEncoder

class RAM(nn.Module):
    """Reconstruction-Anchored Diffusion Model (paper's RAM).

    Two branches sharing the diffusion decoder D (MDMDecoder):
      - motion reconstruction: E_m (motion_encoder) -> z_m -> D
      - text-to-motion generation: E_t (text_encoder) -> z_t -> D
    """
    def __init__(self, config, mdm: MDMDecoder, text_encoder: ACTORStyleEncoder, motion_encoder: ACTORStyleEncoder):
        super().__init__()

        self.share_weight = config['share_weight']
        self.use_motion_encoder = config['use_motion_encoder']
        self.motion_decoder = mdm
        if self.use_motion_encoder:
            # self.motion_encoder = mdm if self.share_weight else copy.deepcopy(mdm)
            self.motion_encoder = motion_encoder
        self.text_encoder = text_encoder
        self.vae = config['vae']
        self.fact = config['fact']
        self.cond_mask_prob = self.motion_decoder.cond_mask_prob
        self.cond_mode = self.motion_decoder.cond_mode
        self.translation = self.motion_decoder.translation
        self.njoints = self.motion_decoder.njoints
        self.nfeats = self.motion_decoder.nfeats
        self.data_rep = self.motion_decoder.data_rep
        self.encoder_stop_step = config.get('encoder_stop_step', 9999999)
        self.rot2xyz = self.motion_decoder.rot2xyz
        
        # self.mean_mdm = torch.from_numpy(np.load('dataset/t2m_mean.npy')).to('cuda').to(torch.float32)
        # self.std_mdm = torch.from_numpy(np.load('dataset/t2m_std.npy')).to('cuda').to(torch.float32)
        
        # # 使用torch.load加载.pt文件
        # self.mean_tmr = torch.load('utils/TMR/stats/humanml3d/guoh3dfeats/mean.pt').to('cuda').to(torch.float32)
        # self.std_tmr = torch.load('utils/TMR/stats/humanml3d/guoh3dfeats/std.pt').to('cuda').to(torch.float32)
        
    def _find_encoder(self, inputs, modality):
        assert modality in ["text", "motion", "auto"]

        if modality == "text":
            return self.text_encoder
        elif modality == "motion":
            return self.motion_encoder

        m_nfeats = self.motion_decoder.nfeats
        t_nfeats = self.text_encoder.nfeats

        if m_nfeats == t_nfeats:
            raise ValueError(
                "Cannot automatically find the encoder, as they share the same input space."
            )

        nfeats = inputs["x"].shape[-1]
        
        if nfeats == m_nfeats:
            inputs['mask'] = inputs['m_mask']
            # inputs['x'] = ((inputs['x'].clone() * self.std_mdm + self.mean_mdm) - self.mean_tmr) / self.std_tmr
            return self.motion_encoder, inputs
        elif nfeats == t_nfeats:
            inputs['mask'] = inputs['t_mask']
            return self.text_encoder, inputs
        else:
            raise ValueError("The inputs is not recognized.")
        
    def encode(
        self,
        inputs,
        modality: str = "auto",
        sample_mean: Optional[bool] = None,
        fact: Optional[float] = None,
        return_distribution: bool = False,
    ):
        sample_mean = False if sample_mean is None else sample_mean
        fact = self.fact if fact is None else fact

        # Encode the inputs
        encoder, inputs = self._find_encoder(inputs, modality)
        encoded, motions_enc = encoder(inputs)

        # Sampling
        if self.vae:
            dists = encoded.unbind(1)
            mu, logvar = dists
            if sample_mean:
                latent_vectors = mu
            else:
                # Reparameterization trick
                std = logvar.exp().pow(0.5)
                eps = std.data.new(std.size()).normal_()
                latent_vectors = mu + fact * eps * std
        else:
            dists = None
            # (latent_vectors,) = encoded.unbind(1)
            (latent_vectors,) = encoded.unbind(1)

        if return_distribution:
            return motions_enc, latent_vectors, dists

        return latent_vectors

    # Forward: X => motions
    def train_forward(
        self,
        inputs,
        sample_mean: Optional[bool] = None,
        fact: Optional[float] = None,
        return_all: bool = False,
    ) -> List[Tensor]:
        # Encoding the inputs and sampling if needed
        
        inputs_enc = {'x': inputs['x_enc'], 't': inputs['t_enc'], 't_mask':inputs['t_mask'].squeeze(), 'm_mask': inputs['m_mask'].squeeze()}
        if inputs['step'] < self.encoder_stop_step:
            motions_enc, latent_vectors, distributions = self.encode(
                inputs_enc, sample_mean=sample_mean, fact=fact, return_distribution=True
            )
        else:
            with torch.no_grad():
                motions_enc, latent_vectors, distributions = self.encode(
                    inputs_enc, sample_mean=sample_mean, fact=fact, return_distribution=True
                )
        # Decoding the latent vector: generating motions
        inputs_dec = {'x': inputs['x_dec'], 't': inputs['t_dec'], 'mask': inputs['m_mask'], 'z': latent_vectors.unsqueeze(1)}
        if 'clip_emb' in inputs.keys():
            inputs_dec['clip_emb'] = inputs['clip_emb']
        latent_dec, motions_dec = self.motion_decoder(inputs_dec)
        
        if motions_enc is not None:
            motions = [motions_enc, motions_dec]
        else:
            motions = motions_dec

        if return_all:
            return motions, latent_vectors, latent_dec, distributions

        return motions
        
    def forward(
        self, x=None, t=None, y=None, train=False, **kwargs
    ):  
        if not train:
            # sampling
            # import pdb;pdb.set_trace()
            inputs_dec = {'x': x.squeeze(2).permute(0,2,1), 't': t, 'mask': y['mask'].squeeze(1).permute(0,2,1), 'z': y['latent'].unsqueeze(1), 'uncond': y.get('uncond', False)}
            _, motions_dec = self.motion_decoder(inputs_dec)
            motions_dec = motions_dec.permute(0,2,1).unsqueeze(2)
            return motions_dec
        else:
            return self.train_forward(**kwargs)
        
        
def length_to_mask(length: List[int], device: torch.device = None) -> Tensor:
    if device is None:
        device = "cpu"

    if isinstance(length, list):
        length = torch.tensor(length, device=device)

    max_len = max(length)
    mask = torch.arange(max_len, device=device).expand(
        len(length), max_len
    ) < length.unsqueeze(1)
    return mask