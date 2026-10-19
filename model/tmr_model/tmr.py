from typing import Dict, Optional
from torch import Tensor

import torch
import torch.nn as nn
from .temos import TEMOS
from .losses import InfoNCE_with_filtering
from .metrics import all_contrastive_metrics


# x.T will be deprecated in pytorch
def transpose(x):
    return x.permute(*torch.arange(x.ndim - 1, -1, -1))


def get_sim_matrix(x, y):
    x_logits = torch.nn.functional.normalize(x, dim=-1)
    y_logits = torch.nn.functional.normalize(y, dim=-1)
    sim_matrix = x_logits @ transpose(y_logits)
    return sim_matrix


# Scores are between 0 and 1
def get_score_matrix(x, y):
    sim_matrix = get_sim_matrix(x, y)
    scores = sim_matrix / 2 + 0.5
    return scores


class TMR(TEMOS):
    r"""TMR: Text-to-Motion Retrieval
    Using Contrastive 3D Human Motion Synthesis
    Find more information about the model on the following website:
    https://mathis.petrovich.fr/tmr

    Args:
        motion_encoder: a module to encode the input motion features in the latent space (required).
        text_encoder: a module to encode the text embeddings in the latent space (required).
        motion_decoder: a module to decode the latent vector into motion features (required).
        vae: a boolean to make the model probabilistic (required).
        fact: a scaling factor for sampling the VAE (optional).
        sample_mean: sample the mean vector instead of random sampling (optional).
        lmd: dictionary of losses weights (optional).
        lr: learninig rate for the optimizer (optional).
        temperature: temperature of the softmax in the contrastive loss (optional).
        threshold_selfsim: threshold used to filter wrong negatives for the contrastive loss (optional).
        threshold_selfsim_metrics: threshold used to filter wrong negatives for the metrics (optional).
    """

    def __init__(
        self,
        motion_encoder: nn.Module,
        text_encoder: nn.Module,
        motion_decoder: nn.Module,
        vae: bool,
        fact: Optional[float] = None,
        sample_mean: Optional[bool] = False,
        lmd: Dict = {"recons": 1.0, "latent": 1.0e-5, "kl": 1.0e-5, "contrastive": 0.1},
        lr: float = 1e-4,
        temperature: float = 0.7,
        threshold_selfsim: float = 0.80,
        threshold_selfsim_metrics: float = 0.95,
    ) -> None:
        # Initialize module like TEMOS
        super().__init__(
            motion_encoder=motion_encoder,
            text_encoder=text_encoder,
            motion_decoder=motion_decoder,
            vae=vae,
            fact=fact,
            sample_mean=sample_mean,
            lmd=lmd,
            lr=lr,
        )

        # adding the contrastive loss
        self.contrastive_loss_fn = InfoNCE_with_filtering(
            temperature=temperature, threshold_selfsim=threshold_selfsim
        )
        self.threshold_selfsim_metrics = threshold_selfsim_metrics

        # store validation values to compute retrieval metrics
        # on the whole validation set
        self.validation_step_t_latents = []
        self.validation_step_m_latents = []
        self.validation_step_sent_emb = []
        
        self.compute_loss = self.compute_loss_origin

    def compute_loss_origin(self, batch: Dict, return_all=False) -> Dict:
        
        text_x_dict = batch["text_x_dict"]
        motion_x_dict = batch["motion_x_dict"]

        mask = motion_x_dict["mask"]
        ref_motions = motion_x_dict["x"]

        # sentence embeddings
        sent_emb = batch["sent_emb"]

        # text -> motion
        t_motions, t_latents, t_dists = self(text_x_dict, mask=mask, return_all=True)

        # motion -> motion
        m_motions, m_latents, m_dists = self(motion_x_dict, mask=mask, return_all=True)

        # Store all losses
        losses = {}

        # Reconstructions losses
        # fmt: off
        losses["recons"] = (
            + self.reconstruction_loss_fn(t_motions, ref_motions) # text -> motion
            + self.reconstruction_loss_fn(m_motions, ref_motions) # motion -> motion
        )
        # fmt: on

        # VAE losses
        if self.vae:
            # Create a centred normal distribution to compare with
            # logvar = 0 -> std = 1
            ref_mus = torch.zeros_like(m_dists[0])
            ref_logvar = torch.zeros_like(m_dists[1])
            ref_dists = (ref_mus, ref_logvar)

            losses["kl"] = (
                self.kl_loss_fn(t_dists, m_dists)  # text_to_motion
                + self.kl_loss_fn(m_dists, t_dists)  # motion_to_text
                + self.kl_loss_fn(m_dists, ref_dists)  # motion
                + self.kl_loss_fn(t_dists, ref_dists)  # text
            )

        # Latent manifold loss
        losses["latent"] = self.latent_loss_fn(t_latents, m_latents)

        # TMR: adding the contrastive loss
        losses["contrastive"] = self.contrastive_loss_fn(t_latents, m_latents, sent_emb)

        # Weighted average of the losses
        losses["loss"] = sum(
            self.lmd[x] * val for x, val in losses.items() if x in self.lmd
        )
        
        # Used for the validation step
        if return_all:
            return losses, t_latents, m_latents

        return losses



    def validation_step(self, batch: Dict, batch_idx: int) -> Tensor:
        bs = len(batch["motion_x_dict"]["x"])
        losses, t_latents, m_latents = self.compute_loss_origin(batch, return_all=True)

        # Store the latent vectors
        self.validation_step_t_latents.append(t_latents)
        self.validation_step_m_latents.append(m_latents)
        self.validation_step_sent_emb.append(batch["sent_emb"])

        for loss_name in sorted(losses):
            loss_val = losses[loss_name]
            self.log(
                f"val_{loss_name}",
                loss_val,
                on_epoch=True,
                on_step=True,
                batch_size=bs,
            )

        return losses["loss"]

    def on_validation_epoch_end(self):
        # Compute contrastive metrics on the whole batch
        t_latents = torch.cat(self.validation_step_t_latents)
        m_latents = torch.cat(self.validation_step_m_latents)
        sent_emb = torch.cat(self.validation_step_sent_emb)

        # Compute the similarity matrix
        sim_matrix = get_sim_matrix(t_latents, m_latents).cpu().numpy()

        contrastive_metrics = all_contrastive_metrics(
            sim_matrix,
            emb=sent_emb.cpu().numpy(),
            threshold=self.threshold_selfsim_metrics,
        )

        for loss_name in sorted(contrastive_metrics):
            loss_val = contrastive_metrics[loss_name]
            self.log(
                f"val_{loss_name}_epoch",
                loss_val,
                on_epoch=True,
                on_step=False,
            )

        self.validation_step_t_latents.clear()
        self.validation_step_m_latents.clear()
        self.validation_step_sent_emb.clear()
        
    def compute_loss_pair_v2(self, batch: Dict, return_all=False) -> Dict:
        text_x_dict = batch["text_x_dict"]
        motion_x_dict = batch["motion_x_dict"]

        mask = motion_x_dict["mask"]
        ref_motions_single = motion_x_dict["x"][:, 0]
        ref_motions_all = motion_x_dict["x"].reshape(-1, 200, 263)
        motion_x_dict["x"] = ref_motions_all

        # sentence embeddings
        sent_emb = batch["sent_emb"]
        
        sent_emb_repeat = sent_emb.unsqueeze(1).repeat(1, 4, 1).reshape(ref_motions_all.shape[0], -1)

        # text -> motion
        t_motions, t_latents, t_dists = self(text_x_dict, mask=mask, return_all=True)
        
        mask = mask.unsqueeze(1).repeat(1, 4, 1).reshape(-1, 200)
        motion_x_dict['mask'] = mask
        # motion -> motion
        m_motions, m_latents, m_dists = self(motion_x_dict, mask=mask, return_all=True)

        # Store all losses
        losses = {}

        # Reconstructions losses
        # fmt: off
        losses["recons"] = (
            + self.reconstruction_loss_fn(t_motions, ref_motions_single) # text -> motion
            + self.reconstruction_loss_fn(m_motions, ref_motions_all) # motion -> motion
        )
        # fmt: on

        # VAE losses
        if self.vae:
            # Create a centred normal distribution to compare with
            # logvar = 0 -> std = 1
            ref_mus = torch.zeros_like(m_dists[0])
            ref_logvar = torch.zeros_like(m_dists[1])
            m_ref_dists = (ref_mus, ref_logvar)
            ref_mus = torch.zeros_like(t_dists[0])
            ref_logvar = torch.zeros_like(t_dists[1])
            t_ref_dists = (ref_mus, ref_logvar)

            losses["kl"] = (
                self.kl_loss_fn(t_dists, m_dists)  # text_to_motion
                + self.kl_loss_fn(m_dists, t_dists)  # motion_to_text
                + self.kl_loss_fn(m_dists, m_ref_dists)  # motion
                + self.kl_loss_fn(t_dists, t_ref_dists)  # text
            )

        # Latent manifold loss
        losses["latent"] = self.latent_loss_fn(t_latents, m_latents.reshape(-1, 4, 256).mean(1))

        # TMR: adding the contrastive loss
        losses["contrastive"] = self.contrastive_loss_fn(t_latents, m_latents, sent_emb, sent_emb_repeat)

        # Weighted average of the losses
        losses["loss"] = sum(
            self.lmd[x] * val for x, val in losses.items() if x in self.lmd
        )

        # Used for the validation step
        if return_all:
            return losses, t_latents, m_latents

        return losses


    def expand_mask_for_multi_text(self, motion_mask, lengths_per_sample):
        """
        根据lengths_per_sample扩展motion的mask以匹配text的数量
        
        Args:
            motion_mask: [batch_size, seq_len] motion的mask
            lengths_per_sample: [batch_size] 每个样本对应的文本数量
            
        Returns:
            expanded_mask: [total_texts, seq_len] 扩展后的mask
        """
        expanded_masks = []
        
        for sample_idx, num_texts in enumerate(lengths_per_sample):
            # 获取当前样本的mask
            sample_mask = motion_mask[sample_idx]  # [seq_len]
            
            # 将这个mask复制num_texts次
            for _ in range(num_texts):
                expanded_masks.append(sample_mask)
        
        # 堆叠所有mask
        expanded_mask = torch.stack(expanded_masks, dim=0)  # [total_texts, seq_len]
        
        return expanded_mask
