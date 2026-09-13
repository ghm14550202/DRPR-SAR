"""基于 Rotation Trick 的 VQ-VAE, 用于替换 CD-VAE 中的高斯 VAE.

设计原则: encoder / decoder / xi_bn / classifier 与 networks_V1.adv_vae_SAR.CVAE_SAR_Res
完全一致 (classifier 为 resnet50), 只把 "fc11/fc12 -> reparameterize -> fc21" 这条
高斯采样瓶颈换成 "pre_quant_proj -> 空间 VQ (+ rotation trick) -> post_quant_proj".

Rotation trick 参考 rotation_trick-main/src/models/vq_vae.py:
    - VQ 在每个空间位置上独立做 (accept_image_fmap), 展平成 (b h w) c 后查表;
    - 用 Householder 旋转替代 STE, 让梯度沿 e -> q 的旋转方向传播;
    - 缩放因子 ||q|| / ||e|| 视为常数 (detach).
"""

from __future__ import annotations

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F

from .adv_vae_SAR import AbstractAutoEncoder, ResBlock
from .resnet import resnet50


class VectorQuantizeEMA(nn.Module):
    """EMA codebook 的向量量化层, 输入输出均为 [B, C, H, W].

    注意: codebook 用 buffer + EMA 更新, 在 DataParallel 下每个 replica 的 buffer
    改动会被丢弃, 导致 codebook 学不动. 需要多卡时请改用 DistributedDataParallel,
    或把 learnable_codebook 打开改成梯度更新.
    """

    def __init__(self, dim, codebook_size, commitment_weight=1.0, decay=0.8,
                 eps=1e-5, threshold_ema_dead_code=0):
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        self.commitment_weight = commitment_weight
        self.decay = decay
        self.eps = eps
        self.threshold_ema_dead_code = threshold_ema_dead_code

        embed = torch.empty(codebook_size, dim)
        nn.init.kaiming_uniform_(embed)
        self.register_buffer('embedding', embed)
        self.register_buffer('cluster_size', torch.zeros(codebook_size))
        self.register_buffer('embed_avg', embed.clone())

    @torch.no_grad()
    def _ema_update(self, flattened, indices):
        if not self.training:
            return

        # autocast (AMP) 下 flattened 是 fp16, 而 codebook buffer 是 fp32。
        # EMA 累积必须在 fp32 下做, 否则既有 dtype 不匹配的报错, 也会丢精度。
        flattened = flattened.to(self.embedding.dtype)

        one_hot = F.one_hot(indices, self.codebook_size).type(flattened.dtype)
        cluster_size = one_hot.sum(dim=0)
        embed_sum = one_hot.t() @ flattened

        self.cluster_size.mul_(self.decay).add_(cluster_size, alpha=1.0 - self.decay)
        self.embed_avg.mul_(self.decay).add_(embed_sum, alpha=1.0 - self.decay)

        smoothed_cluster_size = (
            (self.cluster_size + self.eps)
            / (self.cluster_size.sum() + self.codebook_size * self.eps)
            * self.cluster_size.sum()
        )
        self.embedding.copy_(self.embed_avg / smoothed_cluster_size.unsqueeze(1).clamp(min=self.eps))

        if self.threshold_ema_dead_code <= 0:
            return

        # 复活长期没被选中的 code, 避免 codebook collapse
        expired = self.cluster_size < self.threshold_ema_dead_code
        num_expired = int(expired.sum().item())
        if num_expired == 0 or flattened.shape[0] == 0:
            return

        rand_idx = torch.randint(0, flattened.shape[0], (num_expired,), device=flattened.device)
        replacement = flattened[rand_idx]
        self.embedding[expired] = replacement
        self.embed_avg[expired] = replacement * self.threshold_ema_dead_code
        self.cluster_size[expired] = float(self.threshold_ema_dead_code)

    def forward(self, x):
        b, c, h, w = x.shape
        flattened = rearrange(x, 'b c h w -> (b h w) c')

        d = (
            flattened.pow(2).sum(dim=1, keepdim=True)
            + self.embedding.pow(2).sum(dim=1)
            - 2 * flattened @ self.embedding.t()
        )
        indices = d.argmin(dim=1)
        z_q = self.embedding[indices]

        self._ema_update(flattened.detach(), indices.detach())

        # commitment loss: 只约束 encoder 靠近 codebook (codebook 由 EMA 更新)
        # 返回 shape [1] 而非 0 维标量: DataParallel 的 gather 无法拼接 0 维张量
        commit_loss = self.commitment_weight * F.mse_loss(flattened, z_q.detach())

        z_q = rearrange(z_q, '(b h w) c -> b c h w', b=b, h=h, w=w)
        indices = rearrange(indices, '(b h w) -> b h w', b=b, h=h, w=w)
        return z_q, indices, commit_loss.view(1)


class RotVQVAE_SAR_Res(AbstractAutoEncoder):
    """与 CVAE_SAR_Res 接口对齐的 rotation-trick VQ-VAE.

    forward 返回值刻意与 CVAE_SAR_Res 保持相同的元数, 使 adv_train 侧无需改动:
        with_classifier=False -> (xi, commit_loss, None)
        with_classifier=True  -> (out_g, out_r, hi, xi, commit_loss, None)
        classifier_mode="residual" 时 out_g 为 None，classifier 只接收 x-xi；
        默认 classifier_mode="concat" 保留旧接口。
    其中第 5 个位置由 logvar/mu 换成 commit_loss, 第 6 个位置恒为 None.
    """

    def __init__(self, d, z, num_embeddings=512, commitment_weight=1.0, decay=0.8,
                 threshold_ema_dead_code=0, with_classifier=True, num_classes=10,
                 num_channels=1, classifier_mode="concat", **kwargs):
        super(RotVQVAE_SAR_Res, self).__init__()

        # === encoder: 与 CVAE_SAR_Res 逐层一致 ===
        self.encoder = nn.Sequential(
            nn.Conv2d(num_channels, d // 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(d // 2, d, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d),
            ResBlock(d, d, bn=True),
        )

        # === decoder: 与 CVAE_SAR_Res 逐层一致 ===
        self.decoder = nn.Sequential(
            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d),
            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d),
            nn.ConvTranspose2d(d, d // 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d // 2),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(d // 2, 1, kernel_size=4, stride=2, padding=1, bias=False),
        )

        self.xi_bn = nn.BatchNorm2d(1)

        self.f = 32  # 128 / 4, encoder 两次 stride=2 后的空间尺寸
        self.d = d
        self.z = z  # codebook 向量维度

        # 逐像素 1x1 投影进出 codebook (等价于参考实现里对 (b h w) c 做 Linear)
        self.pre_quant_proj = nn.Conv2d(d, self.z, kernel_size=1)
        self.post_quant_proj = nn.Conv2d(self.z, d, kernel_size=1)

        self.vq = VectorQuantizeEMA(
            dim=self.z,
            codebook_size=num_embeddings,
            commitment_weight=commitment_weight,
            decay=decay,
            threshold_ema_dead_code=threshold_ema_dead_code,
        )

        self.with_classifier = with_classifier
        self.classifier_mode = classifier_mode
        if self.with_classifier:
            self.classifier = resnet50(pretrained=False, num_classes=num_classes)

    @staticmethod
    def get_very_efficient_rotation(u, q, e):
        """Householder 形式的高效旋转, 把 e 旋转到 u -> q 的方向上.

        u, q: [N, C] 已归一化; e: [N, 1, C]. 返回 [N, 1, C].
        """
        w = ((u + q) / (torch.norm(u + q, dim=1, keepdim=True) + 1e-6)).detach()
        e = e - 2 * torch.bmm(torch.bmm(e, w.unsqueeze(-1)), w.unsqueeze(1)) \
            + 2 * torch.bmm(torch.bmm(e, u.unsqueeze(-1).detach()), q.unsqueeze(1).detach())
        return e

    def encode(self, x):
        h = self.encoder(x)
        z_e = self.pre_quant_proj(h)
        return h, z_e

    def decode(self, z_q):
        h = self.post_quant_proj(z_q)
        h = self.decoder(h)
        return torch.tanh(h)

    def quantize(self, z_e, rot=True):
        """VQ 查表, 并用 rotation trick 替代 STE 传梯度."""
        z_q, indices, commit_loss = self.vq(z_e)

        if not rot:
            # 退化为标准 STE
            z_q = z_e + (z_q - z_e).detach()
            return z_q, indices, commit_loss

        b, c, h, w = z_e.shape
        e = rearrange(z_e, 'b c h w -> (b h w) c')
        q = rearrange(z_q, 'b c h w -> (b h w) c')

        e_norm = torch.norm(e, dim=1, keepdim=True) + 1e-6
        q_norm = torch.norm(q, dim=1, keepdim=True) + 1e-6

        rotated = self.get_very_efficient_rotation(e / e_norm, q / q_norm, e.unsqueeze(1)).squeeze(1)
        # 缩放到 ||q||, 缩放因子视为常数
        rotated = rotated * (q_norm / e_norm).detach()

        z_q = rearrange(rotated, '(b h w) c -> b c h w', b=b, h=h, w=w)
        return z_q, indices, commit_loss

    def forward(self, x, rot=True):
        _, z_e = self.encode(x)
        z_q, _, commit_loss = self.quantize(z_e, rot=rot)
        xi = self.xi_bn(self.decode(z_q))

        if self.with_classifier:
            if self.classifier_mode == "residual":
                # 第一阶段的目标是让分类器只学习残差 x-xi；xi 不进入该分类器。
                out_r = self.classifier(x - xi)
                out_g = None
            elif self.classifier_mode == "concat":
                out = self.classifier(torch.cat((xi, x - xi), dim=0))
                out_g = out[0:x.size(0)]
                out_r = out[x.size(0):]
            else:
                raise ValueError(f"Unsupported classifier_mode: {self.classifier_mode}")
            # hi 位置返回量化后的隐变量, 与 CVAE_SAR_preres 的元数对齐
            return out_g, out_r, z_q, xi, commit_loss, None
        return xi, commit_loss, None
