import torch
import torch.nn as nn
import torch.nn.functional as F


# For reference
# https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
# https://pytorch.org/docs/stable/_modules/torch/distributions/kl.html#kl_divergence
class KLLoss:
    def __call__(self, q, p):
        mu_q, logvar_q = q
        mu_p, logvar_p = p

        log_var_ratio = logvar_q - logvar_p
        t1 = (mu_p - mu_q).pow(2) / logvar_p.exp()
        div = 0.5 * (log_var_ratio.exp() + t1 - 1 - log_var_ratio)
        return div.mean()

    def __repr__(self):
        return "KLLoss()"


class InfoNCE_with_filtering:
    def __init__(self, temperature=0.7, threshold_selfsim=0.8):
        self.temperature = temperature
        self.threshold_selfsim = threshold_selfsim
        self.smoothing_factor = -(1e10)

    def get_sim_matrix(self, x, y):
        x_logits = torch.nn.functional.normalize(x, dim=-1)
        y_logits = torch.nn.functional.normalize(y, dim=-1)
        sim_matrix = x_logits @ y_logits.T
        return sim_matrix

    def __call__(self, x, y, sent_emb=None):
        bs, device = len(x), x.device
        sim_matrix = self.get_sim_matrix(x, y) / self.temperature

        if sent_emb is not None and self.threshold_selfsim:
            # put the threshold value between -1 and 1
            real_threshold_selfsim = 2 * self.threshold_selfsim - 1
            # Filtering too close values
            # mask them by putting -inf in the sim_matrix
            selfsim = sent_emb @ sent_emb.T
            selfsim_nodiag = selfsim - selfsim.diag().diag()
            idx = torch.where(selfsim_nodiag > real_threshold_selfsim)
            sim_matrix[idx] = -torch.inf
  
        labels = torch.arange(bs, device=device)
        
        # num_classes = bs // 4
        # labels = torch.repeat_interleave(torch.arange(num_classes, device=device), 4)

        total_loss = (
            F.cross_entropy(sim_matrix, labels) + F.cross_entropy(sim_matrix.T, labels)
        ) / 2

        return total_loss

    def __repr__(self):
        return f"Constrastive(temp={self.temp})"
    
    
def block_diagonal_gradients(matrix):
    """
    保持矩阵对角线元素的值不变，但阻止梯度通过对角线元素
    
    参数:
    matrix: 输入的n×n矩阵
    
    返回:
    处理后的矩阵
    """
    # 创建一个与输入矩阵相同的新矩阵
    processed = matrix.clone()
    
    # 创建对角线掩码
    mask = torch.eye(matrix.size(0), dtype=torch.bool, device=matrix.device)
    
    # 提取对角线元素并分离
    diag_elements = matrix[mask].detach()
    
    # 将分离后的对角线元素放回矩阵
    processed[mask] = diag_elements
    
    return processed


class SelfRegularizationLoss:
    """Self-regularization loss L_sr (paper Eq.5).

    A cross-entropy over the motion latent space that treats only identical
    indices as positive pairs, pushing motion latents apart for better
    separability. Beyond Eq.5, this implementation additionally (a) masks out
    near-duplicate captions via `threshold_selfsim` on `sent_emb`, and
    (b) blocks gradients through the diagonal (self-similarity) terms with
    `block_diagonal_gradients`. Both are implementation details not expanded in
    the paper; kept as-is to preserve the released model's behavior.
    """
    def __init__(self, temperature=0.7, threshold_selfsim=0.8):
        self.temperature = temperature
        self.threshold_selfsim = threshold_selfsim
        self.smoothing_factor = -(1e10)

    def get_sim_matrix(self, x,):
        x_logits = torch.nn.functional.normalize(x, dim=-1)
        sim_matrix = x_logits @ x_logits.T
        return sim_matrix

    def __call__(self, x, sent_emb=None):
        bs, device = len(x), x.device
        sim_matrix = self.get_sim_matrix(x) / self.temperature

        if sent_emb is not None and self.threshold_selfsim:
            # put the threshold value between -1 and 1
            real_threshold_selfsim = 2 * self.threshold_selfsim - 1
            # Filtering too close values
            # mask them by putting -inf in the sim_matrix
            selfsim = sent_emb @ sent_emb.T
            # selfsim = sim_matrix.detach()
            selfsim_nodiag = selfsim - selfsim.diag().diag()
            idx = torch.where(selfsim_nodiag > real_threshold_selfsim)
            sim_matrix[idx] = -torch.inf
  
        labels = torch.arange(bs, device=device)
        
        sim_matrix = block_diagonal_gradients(sim_matrix)
        
        return F.cross_entropy(sim_matrix, labels)

    def __repr__(self):
        return f"Constrastive(temp={self.temp})"
    
    
def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def hard_example_mining(dist_mat, labels_1, labels_2=None, return_inds=False):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      labels: pytorch LongTensor, with shape [N]
      return_inds: whether to return the indices. Save time if `False`(?)
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """

    assert len(dist_mat.size()) == 2
    # assert dist_mat.size(0) == dist_mat.size(1)
    N, M = dist_mat.size(0), dist_mat.size(1)
    
    if labels_2 is None:
        # shape [N, N]
        is_pos = labels_1.expand(N, N).eq(labels_1.expand(N, N).t())
        is_neg = labels_1.expand(N, N).ne(labels_1.expand(N, N).t())
    else:
        is_pos = labels_1.expand(N, M).eq(labels_2.expand(M, N).t())
        is_neg = labels_1.expand(N, M).ne(labels_2.expand(M, N).t())

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N, 1]
    dist_ap, relative_p_inds = torch.max(
        dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N, 1]
    dist_an, relative_n_inds = torch.min(
        dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
    # shape [N]
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    # if return_inds:
    #     # shape [N, N]
    #     ind = (labels.new().resize_as_(labels)
    #            .copy_(torch.arange(0, N).long())
    #            .unsqueeze(0).expand(N, N))
    #     # shape [N, 1]
    #     p_inds = torch.gather(
    #         ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
    #     n_inds = torch.gather(
    #         ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
    #     # shape [N]
    #     p_inds = p_inds.squeeze(1)
    #     n_inds = n_inds.squeeze(1)
    #     return dist_ap, dist_an, p_inds, n_inds

    return dist_ap, dist_an


class TripletLoss(object):
    """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
    Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
    Loss for Person Re-Identification'."""

    def __init__(self, margin=None):
        self.margin = margin
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, global_feat, labels, normalize_feature=False):
        if normalize_feature:
            global_feat = normalize(global_feat, axis=-1)
        dist_mat = euclidean_dist(global_feat, global_feat)
        dist_ap, dist_an = hard_example_mining(
            dist_mat, labels)
        y = dist_an.new().resize_as_(dist_an).fill_(1)
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
        return loss, dist_ap, dist_an


class TripletLoss_two_vector(object):
    """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
    Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
    Loss for Person Re-Identification'."""

    def __init__(self, margin=None):
        self.margin = margin
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, feat_1, feat_2, labels_1, labels_2=None, normalize_feature=False):
        if normalize_feature:
            feat_1 = normalize(feat_1, axis=-1)
            feat_2 = normalize(feat_2, axis=-1)
        dist_mat = euclidean_dist(feat_1, feat_2)
        dist_ap, dist_an = hard_example_mining(
            dist_mat, labels_1, labels_2)
        y = dist_an.new().resize_as_(dist_an).fill_(1)
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
        return loss, dist_ap, dist_an

if __name__ == '__main__':
    # global_feat = torch.randn([64, 128]).to('cuda')
    # label = torch.arange(16).repeat_interleave(4).to('cuda')
    
    # loss_fn = TripletLoss(0.3)
    # output = loss_fn(global_feat, label)
    
    feat_1 = torch.randn([32, 128]).to('cuda')
    feat_2 = torch.randn([128, 128]).to('cuda')
    label_1= torch.arange(32).view(-1,1).to('cuda')
    label_2 = torch.arange(32).repeat_interleave(4).view(-1,1).to('cuda')
    
    loss_fn = TripletLoss_two_vector(0.3)
    output = loss_fn(feat_1, feat_2, label_1, label_2)
    
    
    
def compute_multi_text_reconstruction_loss(t_motions, ref_motions, lengths_per_sample, reduction="mean"):
    """
    计算多文本重建损失
    
    Args:
        t_motions: 从文本生成的动作 [total_texts, seq_len, motion_dim]
        ref_motions: 参考动作 [batch_size, seq_len, motion_dim] 
        lengths_per_sample: 每个样本包含的文本数量 [batch_size]
        reduction: 损失聚合方式 ("mean", "sum", "none")
    
    Returns:
        loss: 重建损失
    """
    losses = []
    text_idx = 0
    
    # 为每个样本计算损失
    for sample_idx, num_texts in enumerate(lengths_per_sample):
        # 获取当前样本的参考动作
        ref_motion = ref_motions[sample_idx]  # [seq_len, motion_dim]
        
        # 获取当前样本对应的所有文本生成的动作
        sample_t_motions = t_motions[text_idx:text_idx + num_texts]  # [num_texts, seq_len, motion_dim]
        
        # 计算每个文本生成动作与参考动作的损失
        sample_losses = []
        for text_motion in sample_t_motions:
            loss = torch.nn.functional.smooth_l1_loss(text_motion, ref_motion, reduction="mean")
            sample_losses.append(loss)
        
        # 对当前样本的多个文本损失进行聚合（可以选择平均或最小值）
        if len(sample_losses) > 1:
            # 方案1: 取平均值
            sample_loss = torch.stack(sample_losses).mean()
            # 方案2: 取最小值（最好的重建）
            # sample_loss = torch.stack(sample_losses).min()
        else:
            sample_loss = sample_losses[0]
        
        losses.append(sample_loss)
        text_idx += num_texts
    
    # 聚合所有样本的损失
    if reduction == "mean":
        return torch.stack(losses).mean()
    elif reduction == "sum":
        return torch.stack(losses).sum()
    else:  # reduction == "none"
        return torch.stack(losses)
    
    
def compute_multi_text_reconstruction_loss_v2(t_motions, ref_motions, lengths_per_sample):
    """简化版本 - 直接使用expand操作"""
    expanded_ref_motions = []
    
    for sample_idx, num_texts in enumerate(lengths_per_sample):
        # 扩展当前样本的参考动作
        ref_motion = ref_motions[sample_idx:sample_idx+1]  # [1, seq_len, motion_dim]
        expanded_motion = ref_motion.expand(num_texts, -1, -1)  # [num_texts, seq_len, motion_dim]
        expanded_ref_motions.append(expanded_motion)
    
    # 连接所有扩展的参考动作
    expanded_ref_motions = torch.cat(expanded_ref_motions, dim=0)  # [total_texts, seq_len, motion_dim]
    
    # 计算重建损失
    return torch.nn.functional.smooth_l1_loss(t_motions, expanded_ref_motions, reduction="mean")



class InfoNCE_with_filtering_multi_text:
    def __init__(self, temperature=0.7, threshold_selfsim=0.8):
        self.temperature = temperature
        self.threshold_selfsim = threshold_selfsim
        self.smoothing_factor = -(1e10)

    def get_sim_matrix(self, x, y):
        x_logits = torch.nn.functional.normalize(x, dim=-1)
        y_logits = torch.nn.functional.normalize(y, dim=-1)
        sim_matrix = x_logits @ y_logits.T
        return sim_matrix

    def create_labels(self, lengths_per_sample, device):
        """
        创建标签矩阵，处理一对多关系
        
        Args:
            lengths_per_sample: 每个样本包含的文本数量 [batch_size]
            device: 设备
            
        Returns:
            text2motion_labels: text到motion的标签 [total_texts]
            motion2text_mask: motion到text的掩码 [batch_size, total_texts]
        """
        total_texts = sum(lengths_per_sample)
        batch_size = len(lengths_per_sample)
        
        # text2motion标签：每个text对应哪个motion
        text2motion_labels = []
        text_idx = 0
        for motion_idx, num_texts in enumerate(lengths_per_sample):
            text2motion_labels.extend([motion_idx] * num_texts)
        text2motion_labels = torch.tensor(text2motion_labels, device=device)
        
        # motion2text掩码：每个motion对应哪些text（用于多标签）
        motion2text_mask = torch.zeros(batch_size, total_texts, device=device)
        text_idx = 0
        for motion_idx, num_texts in enumerate(lengths_per_sample):
            motion2text_mask[motion_idx, text_idx:text_idx + num_texts] = 1.0
            text_idx += num_texts
            
        return text2motion_labels, motion2text_mask

    def multi_label_cross_entropy(self, logits, targets_mask):
        """
        计算多标签交叉熵损失
        
        Args:
            logits: [batch_size, total_texts]
            targets_mask: [batch_size, total_texts] 二值掩码，1表示正样本
            
        Returns:
            loss: 多标签对比损失
        """
        # 对每个motion，计算其与所有text的对比损失
        losses = []
        
        for i in range(logits.size(0)):
            motion_logits = logits[i]  # [total_texts]
            motion_targets = targets_mask[i]  # [total_texts]
            
            # 找到正样本和负样本
            positive_indices = torch.where(motion_targets == 1)[0]
            negative_indices = torch.where(motion_targets == 0)[0]
            
            if len(positive_indices) == 0:
                continue
                
            # 计算正样本的平均log概率
            positive_logits = motion_logits[positive_indices]
            all_logits = motion_logits
            
            # 对于每个正样本，计算InfoNCE损失
            sample_losses = []
            for pos_idx in positive_indices:
                # 计算softmax，分母包含所有样本
                numerator = torch.exp(motion_logits[pos_idx])
                denominator = torch.exp(all_logits).sum()
                loss = -torch.log(numerator / denominator)
                sample_losses.append(loss)
            
            # 对当前motion的所有正样本损失取平均
            motion_loss = torch.stack(sample_losses).mean()
            losses.append(motion_loss)
        
        return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=logits.device)
    
    def multi_label_cross_entropy_simplified(self, logits, targets_mask):
        """
        简化的多标签InfoNCE损失
        
        Args:
            logits: [batch_size, total_texts] (已经除以temperature)
            targets_mask: [batch_size, total_texts] 二值掩码
        """
        # 计算softmax概率
        log_probs = F.log_softmax(logits, dim=-1)  # [batch_size, total_texts]
        
        # 对于每个motion，计算其正样本的平均负对数似然
        # 只在正样本位置计算损失
        positive_log_probs = log_probs * targets_mask  # 只保留正样本的log概率
        
        # 计算每个motion的正样本数量
        num_positives = targets_mask.sum(dim=-1, keepdim=True)  # [batch_size, 1]
        
        # 避免除零
        num_positives = torch.clamp(num_positives, min=1)
        
        # 计算平均负对数似然
        motion_losses = -positive_log_probs.sum(dim=-1) / num_positives.squeeze(-1)
        
        return motion_losses.mean()


    def __call__(self, t_latents, m_latents, sent_emb, lengths_per_sample):
        """
        计算多文本对比损失
        
        Args:
            t_latents: 文本潜在表示 [total_texts, latent_dim]
            m_latents: 动作潜在表示 [batch_size, latent_dim]
            sent_emb: 句子嵌入 [total_texts, emb_dim]
            lengths_per_sample: 每个样本的文本数量 [batch_size]
        """
        device = t_latents.device
        total_texts = t_latents.size(0)
        batch_size = m_latents.size(0)
        
        # 计算相似度矩阵
        sim_matrix = self.get_sim_matrix(t_latents, m_latents) / self.temperature  # [total_texts, batch_size]
        
        # 应用句子嵌入过滤（如果提供）
        # if sent_emb is not None and self.threshold_selfsim:
        #     real_threshold_selfsim = 2 * self.threshold_selfsim - 1
        #     selfsim = sent_emb @ sent_emb.T  # [total_texts, total_texts]
        #     selfsim_nodiag = selfsim - selfsim.diag().diag()
            
        #     # 找到相似度过高的文本对
        #     similar_text_pairs = torch.where(selfsim_nodiag > real_threshold_selfsim)
            
        #     # 对于相似的文本，如果它们对应不同的motion，则在sim_matrix中mask掉
        #     text2motion_labels, _ = self.create_labels(lengths_per_sample, device)
            
        #     for i, j in zip(similar_text_pairs[0], similar_text_pairs[1]):
        #         if text2motion_labels[i] != text2motion_labels[j]:
        #             # 如果两个相似文本对应不同motion，则互相mask
        #             motion_i = text2motion_labels[i]
        #             motion_j = text2motion_labels[j]
        #             sim_matrix[i, motion_j] = -torch.inf
        #             sim_matrix[j, motion_i] = -torch.inf
        
        # 创建标签
        text2motion_labels, motion2text_mask = self.create_labels(lengths_per_sample, device)
        
        # 计算text2motion损失（标准交叉熵）
        text2motion_loss = F.cross_entropy(sim_matrix, text2motion_labels)
        
        # 计算motion2text损失（多标签）
        motion2text_loss = self.multi_label_cross_entropy_simplified(sim_matrix.T, motion2text_mask)
        
        # 总损失
        total_loss = (text2motion_loss + motion2text_loss) / 2
        
        return total_loss

    def __repr__(self):
        return f"ContrastiveMultiText(temp={self.temperature})"
    

class CosineSimilarityPenaltyLoss(nn.Module):
    """
    惩罚样本间余弦相似度的损失函数
    
    参数:
        temperature (float): 温度参数，控制损失的尖锐度。τ越小，损失对高相似度越敏感（更尖锐）；τ越大，损失越平缓
    """
    def __init__(self, temperature: float = 0.5, threshold_selfsim=0.8):
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"温度参数必须为正数，当前值: {temperature}")
        self.temperature = temperature

    def forward(self, features: torch.Tensor, sent_emb) -> torch.Tensor:
        """
        计算损失
        
        参数:
            features (torch.Tensor): 输入特征矩阵，形状 [bs, d]，其中bs为批量大小，d为特征维度
            
        返回:
            torch.Tensor: 标量损失值
        """
        batch_size = features.shape[0]
        
        # 基础校验
        if batch_size < 2:
            raise ValueError(f"批量大小至少需要2个样本，当前值: {batch_size}")
        
        # 步骤1：特征归一化（确保计算的是余弦相似度而非点积）
        # 归一化到单位球面（L2范数为1）
        normalized_features = F.normalize(features, dim=1)
        
        # 步骤2：计算全样本对的余弦相似度矩阵
        # 结果形状 [bs, bs]，sim[i][j] = cos_sim(features[i], features[j])
        similarity_matrix = torch.matmul(normalized_features, normalized_features.T)
        
        # 步骤3：构建掩码排除自身相似度（i==j的情况）
        # 生成对角线为True的掩码矩阵
        identity_mask = torch.eye(batch_size, device=similarity_matrix.device, dtype=torch.bool)
        # 将自身相似度位置填充为-∞（确保logsumexp时不会被选中）
        masked_similarity = similarity_matrix.masked_fill(identity_mask, -float('inf'))
        
        # 步骤4：计算带温度的logsumexp
        # 对每个样本i，计算其与所有j≠i样本的相似度的logsumexp（带温度缩放）
        # 结果形状 [bs]，每个元素是log(Σ_{j≠i} exp(sim[i][j]/τ))
        scaled_similarity = masked_similarity / self.temperature
        logsumexp_per_sample = torch.logsumexp(scaled_similarity, dim=1)
        
        # 步骤5：计算平均相似度的对数（标准化到bs-1个样本）
        # 因为每个样本有(bs-1)个其他样本，所以需要减去log(bs-1)得到平均log相似度
        avg_log_similarity = logsumexp_per_sample - torch.log(torch.tensor(
            batch_size - 1, 
            device=similarity_matrix.device, 
            dtype=torch.float32
        ))
        
        # 步骤6：计算最终损失（平均所有样本的平均log相似度）
        # 目标是最小化该损失 → 等价于最小化样本间的平均余弦相似度
        loss = avg_log_similarity.mean()
        
        return loss