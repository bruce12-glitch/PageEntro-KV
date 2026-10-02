import torch
import torch.nn as nn
import torch.nn.functional as F


class PageEntroKVGroupPooler(nn.Module):
    def __init__(self, num_query_heads=32, num_kv_heads=8, block_size=16, tau_g=0.5, alpha=2.0, sink_size=4):
        super().__init__()
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.group_size = num_query_heads // num_kv_heads
        self.block_size = block_size
        self.tau_g = tau_g
        self.alpha = alpha
        self.sink_size = sink_size

    def renyi_entropy(self, attn):
        eps = 1e-12
        if self.alpha == 1.0:
            return -torch.sum(attn * torch.log(attn + eps), dim=-1)
        s = torch.sum(torch.pow(attn, self.alpha), dim=-1)
        return (1.0 / (1.0 - self.alpha)) * torch.log(s + eps)

    @torch.inference_mode()
    def forward(self, attn_weights, num_retained_blocks):
        Bsz, H_Q, T = attn_weights.shape
        sink_mask = torch.zeros(T, dtype=torch.bool, device=attn_weights.device)
        sink_mask[:min(self.sink_size, T)] = True
        attn_nonsink = attn_weights.masked_fill(sink_mask.unsqueeze(0).unsqueeze(0), 0.0)
        attn_nonsink = attn_nonsink / attn_nonsink.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        ent = self.renyi_entropy(attn_nonsink)
        grouped = attn_weights.view(Bsz, self.num_kv_heads, self.group_size, T)
        ent_g = ent.view(Bsz, self.num_kv_heads, self.group_size)
        weights = F.softmax(-ent_g / self.tau_g, dim=-1).unsqueeze(-1)
        pooled = (weights * grouped).sum(dim=2)
        pad = (self.block_size - T % self.block_size) % self.block_size
        if pad:
            pooled = F.pad(pooled, (0, pad))
        nb = pooled.shape[-1] // self.block_size
        pages = pooled.view(Bsz, self.num_kv_heads, nb, self.block_size).max(dim=-1).values
        idx = torch.topk(pages, num_retained_blocks, dim=-1).indices
        mask = torch.zeros_like(pages, dtype=torch.bool)
        mask.scatter_(-1, idx, True)
        return mask
