import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from typing import List, Dict, Optional

from model.rotation2xyz import Rotation2xyz
from model.network.adanorm import AdaNormEncoderLayer, AdaNormEncoder


class MDMDecoder(nn.Module):
    def __init__(self, modeltype, njoints, nfeats, num_actions, translation, pose_rep, glob, glob_rot,
                 latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1,
                 ablation=None, activation="gelu", legacy=False, data_rep='rot6d', dataset='amass', clip_dim=512,
                 arch='trans_enc', emb_trans_dec=False, clip_version=None, config=None, **kargs):
        super().__init__()

        self.legacy = legacy
        self.modeltype = modeltype
        self.njoints = self.nfeats = njoints
        self.num_actions = num_actions
        self.data_rep = data_rep
        self.dataset = dataset

        self.pose_rep = pose_rep
        self.glob = glob
        self.glob_rot = glob_rot
        self.translation = translation

        self.latent_dim = latent_dim

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.ablation = ablation
        self.activation = activation
        self.input_feats = self.njoints

        self.cond_mask_prob = kargs.get('cond_mask_prob', 0.)
        self.mask_frames = kargs.get('mask_frames', False)
        self.arch = arch
        self.gru_emb_dim = self.latent_dim if self.arch == 'gru' else 0
        self.input_process = InputProcess(self.data_rep, self.input_feats+self.gru_emb_dim, self.latent_dim)

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout, max_len=kargs.get('pos_embed_max_len', 5000))
        
        print(f'architecture of MDMDecoder is {self.arch}')
        if self.arch == 'trans_enc':
            seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                                nhead=self.num_heads,
                                                                dim_feedforward=self.ff_size,
                                                                dropout=self.dropout,
                                                                activation=self.activation,
                                                                batch_first=True)

            self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                            num_layers=self.num_layers)
            
        elif self.arch == 'adan_enc':
            print("ADAN_ENC init")
            seqTransEncoderLayer = AdaNormEncoderLayer(d_model=self.latent_dim,
                                                        nhead=self.num_heads,
                                                        dim_feedforward=self.ff_size,
                                                        dropout=self.dropout,
                                                        activation=self.activation,
                                                        batch_first=True)

            self.seqTransEncoder = AdaNormEncoder(seqTransEncoderLayer,
                                                    num_layers=self.num_layers)

        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)
            
        self.vae = config['vae']
        self.cond_mode = config['cond_mode']
        self.cond_type = config['cond_type']
        self.cond_dim = kargs.get('cond_dim', 256)
        self.cond_proj = nn.Linear(self.cond_dim, self.latent_dim)
        self.nbtokens = 2 if self.vae else 1
        self.tokens = nn.Parameter(torch.randn(self.nbtokens, latent_dim))

        self.output_process = OutputProcess(self.data_rep, self.input_feats, self.latent_dim, self.njoints,
                                            self.nfeats)
        
        self.rot2xyz = Rotation2xyz(device='cpu', dataset=self.dataset)



    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters() if not name.startswith('clip_model.')]

    def mask_cond(self, cond, force_mask=False):
        bs, seq_len, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(bs, 1, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond

    def forward(self, inputs):
        """
        x: [batch_size, max_frames, nfeats, ], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        timesteps = inputs['t']
        x = inputs['x']
        bs, nframes, nfeats = x.shape
        time_emb = self.embed_timestep(timesteps).transpose(0, 1)  # [1, bs, d]

        force_mask = inputs.get('uncond', False)
        
        emb = []
        for cond in self.cond_type:
            if cond in inputs.keys():
                if inputs[cond].shape[-1] != self.latent_dim:
                    c = self.cond_proj(inputs[cond])
                else:
                    c = inputs[cond]
                # c = self.cond_proj(inputs[cond])
                emb.append(c)
            else:
                emb.append(torch.zeros_like(time_emb))
                
        emb = torch.cat(emb, dim=1)
        emb = self.mask_cond(emb, force_mask=force_mask)
        
        if self.cond_mode == 'add':
            emb = emb + time_emb.repeat(1, emb.shape[1], 1)
        elif self.cond_mode == 'concat':
            emb = torch.cat([time_emb, emb], dim=1)
        else:
            raise TypeError
        
        tokens = self.tokens.unsqueeze(0).repeat(bs, 1, 1)
        # emb = torch.cat([emb, tokens], dim=1)
        
        if self.arch == 'trans_enc':
            discard_frames = emb.shape[1] + self.nbtokens
        elif self.arch == 'adan_enc':
            discard_frames = self.nbtokens

        x = self.input_process(x)
        x = torch.cat([tokens, x], dim=1)

        # TODO - move to collate
        frames_mask = None
        # is_valid_mask = inputs['mask'].shape[-1] > 1  # Don't use mask with the generate script
        if self.mask_frames: # and is_valid_mask:
            frames_mask = torch.logical_not(inputs['mask'].squeeze(-1)).to(device=x.device)
            step_mask = torch.zeros((bs, discard_frames), dtype=torch.bool, device=x.device)
            frames_mask = torch.cat([step_mask, frames_mask], dim=1)

        if self.arch == 'trans_enc':
            # adding the timestep embed
            xseq = torch.cat((emb, x), axis=1)  # [seqlen+1, bs, d]
            xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
            output = self.seqTransEncoder(xseq, src_key_padding_mask=frames_mask)  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]
        elif self.arch == 'adan_enc':
            xseq = self.sequence_pos_encoder(x)  # [seqlen+1, bs, d]
            output = self.seqTransEncoder(xseq, src_key_padding_mask=frames_mask, t=emb)
            
        output_x = self.output_process(output[:, discard_frames:])  # [bs, njoints, nfeats, nframes]
        
        return output[:, discard_frames-self.nbtokens:discard_frames], output_x
    

    def _apply(self, fn):
        super()._apply(fn)


    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:, :x.shape[1], :]
        return self.dropout(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[:, timesteps])


class InputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)
        if self.data_rep == 'rot_vel':
            self.velEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        bs, nframes, nfeats = x.shape

        if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
            x = self.poseEmbedding(x)  # [seqlen, bs, d]
            return x
        else:
            raise ValueError


class OutputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim, njoints, nfeats):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.njoints = njoints
        self.nfeats = nfeats
        self.poseFinal = nn.Linear(self.latent_dim, self.input_feats)
        if self.data_rep == 'rot_vel':
            self.velFinal = nn.Linear(self.latent_dim, self.input_feats)

    def forward(self, output):
        nframes, bs, d = output.shape
        if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
            output = self.poseFinal(output)  # [seqlen, bs, 150]
        elif self.data_rep == 'rot_vel':
            first_pose = output[[0]]  # [1, bs, d]
            first_pose = self.poseFinal(first_pose)  # [1, bs, 150]
            vel = output[1:]  # [seqlen-1, bs, d]
            vel = self.velFinal(vel)  # [seqlen-1, bs, 150]
            output = torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, 150]
        else:
            raise ValueError
        return output
