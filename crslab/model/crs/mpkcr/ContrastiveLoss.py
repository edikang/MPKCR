import torch
import torch.nn as nn
import torch.nn.functional as F
class ContrastiveLoss(nn.Module):

    def __init__(self, temperature=0.2):
        super().__init__()
        self.temperature = temperature

    def forward(self, anchor, positives):
        """
        anchor: [batch_size, emb_dim]
        positives: List of [batch_size, emb_dim] 各粒度下的 user 表示
        """
        batch_size = anchor.size(0)
        anchor = F.normalize(anchor, dim=1)

        loss = 0.0
        for pos in positives:
            pos = F.normalize(pos, dim=1)
            logits = torch.mm(anchor, pos.T) / self.temperature  # [batch, batch]
            labels = torch.arange(batch_size).to(anchor.device)
            loss += F.cross_entropy(logits, labels)

        return loss / len(positives)  # 对每个粒度求平均
