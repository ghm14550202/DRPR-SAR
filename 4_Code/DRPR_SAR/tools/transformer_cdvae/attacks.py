import torch
import torch.nn as nn
import torch.nn.functional as F


class NoAttack(nn.Module):
    def forward(self, img, labels):
        return img


class MarginLoss(nn.Module):
    def __init__(self, kappa=10.0):
        super().__init__()
        self.kappa = kappa

    def forward(self, logits, labels):
        one_hot = F.one_hot(labels, num_classes=logits.size(1)).bool()
        correct = logits[one_hot]
        wrong = logits.masked_fill(one_hot, float("-inf")).max(dim=1)[0]
        return torch.clamp(correct - wrong + self.kappa, min=0).sum()


class TransformerAttack(nn.Module):
    def __init__(
        self,
        classifier,
        vae,
        eps_max=8 / 255,
        step_size=None,
        num_iterations=10,
        norm="linf",
        rand_init=True,
        loss="margin",
    ):
        super().__init__()
        self.classifier = classifier
        self.vae = vae
        self.eps_max = eps_max
        self.num_iterations = num_iterations
        self.step_size = step_size or eps_max / (num_iterations ** 0.5)
        self.norm = norm
        self.rand_init = rand_init
        self.loss = loss
        self.criterion = MarginLoss() if loss == "margin" else nn.CrossEntropyLoss()

    def _init_delta(self, x):
        if not self.rand_init:
            return torch.zeros_like(x, requires_grad=True)
        if self.norm == "linf":
            delta = torch.empty_like(x).uniform_(-self.eps_max, self.eps_max)
        elif self.norm == "l2":
            delta = torch.randn_like(x)
            flat = delta.flatten(1)
            norm = flat.norm(p=2, dim=1).clamp_min(1e-12)
            delta = delta / norm.view(-1, 1, 1, 1)
            dim = flat.size(1)
            rand_norm = torch.rand(x.size(0), device=x.device).pow(1.0 / dim)
            delta = delta * rand_norm.view(-1, 1, 1, 1) * self.eps_max
        else:
            raise ValueError(f"Unsupported norm: {self.norm}")
        delta = torch.clamp(x + delta, 0.0, 1.0) - x
        delta.requires_grad_()
        return delta

    def forward(self, img, labels):
        img = img.detach()
        delta = self._init_delta(img)

        for _ in range(self.num_iterations):
            gx, _, _, _ = self.vae(img + delta)
            logits = self.classifier(gx)
            loss = self.criterion(logits, labels)
            loss.backward()

            grad = delta.grad.detach()
            if self.norm == "linf":
                delta.data = delta.data + self.step_size * grad.sign()
                delta.data = delta.data.clamp(-self.eps_max, self.eps_max)
            elif self.norm == "l2":
                batch_size = delta.size(0)
                grad_norm = grad.flatten(1).norm(p=2, dim=1).clamp_min(1e-12)
                delta.data = delta.data + self.step_size * grad / grad_norm.view(batch_size, 1, 1, 1)
                delta_norm = delta.data.flatten(1).norm(p=2, dim=1).clamp_min(1e-12)
                scale = torch.clamp(self.eps_max / delta_norm, max=1.0)
                delta.data = delta.data * scale.view(batch_size, 1, 1, 1)

            delta.data = torch.clamp(img + delta.data, 0.0, 1.0) - img
            delta.grad.zero_()

        return torch.clamp(img + delta.detach(), 0.0, 1.0)

