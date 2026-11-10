"""
model.py  —  3D ResNet regressor for phase fractions (project sheet, item 12).

stem conv -> 4 residual stages (32, 64, 128, 256 channels; stride-2 downsampling between)
-> global average pool -> FC -> 3 logits -> softmax  => [fibre, matrix, void], sums to 1.

Global average pooling makes the network work for ANY cube size (32^3, 64^3, 96^3), which
you need for the sub-volume-size experiment.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock3d(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.c1 = nn.Conv3d(cin, cout, 3, stride, 1, bias=False)
        self.n1 = nn.InstanceNorm3d(cout, affine=True)
        self.c2 = nn.Conv3d(cout, cout, 3, 1, 1, bias=False)
        self.n2 = nn.InstanceNorm3d(cout, affine=True)
        self.skip = None
        if stride != 1 or cin != cout:
            self.skip = nn.Sequential(nn.Conv3d(cin, cout, 1, stride, bias=False),
                                      nn.InstanceNorm3d(cout, affine=True))

    def forward(self, x):
        y = F.leaky_relu(self.n1(self.c1(x)), 0.01)
        y = self.n2(self.c2(y))
        return F.leaky_relu(y + (x if self.skip is None else self.skip(x)), 0.01)


class PhaseFractionResNet3D(nn.Module):
    def __init__(self, widths=(32, 64, 128, 256), blocks=(1, 1, 1, 1), n_out=3):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv3d(1, widths[0], 3, 1, 1, bias=False),
                                  nn.InstanceNorm3d(widths[0], affine=True), nn.LeakyReLU(0.01))
        layers, cin = [], widths[0]
        for i, (w, b) in enumerate(zip(widths, blocks)):
            for j in range(b):
                layers.append(ResBlock3d(cin, w, stride=2 if (j == 0 and i > 0) else 1))
                cin = w
        self.body = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Linear(cin, 64), nn.LeakyReLU(0.01), nn.Linear(64, n_out))

    def forward(self, x):                       # x: (B,1,D,H,W)
        h = self.body(self.stem(x))
        h = h.mean(dim=(2, 3, 4))               # global average pool
        return torch.softmax(self.head(h), dim=1)   # (B,3): fibre, matrix, void


def fraction_loss(pred, true, w=(1.0, 1.0, 10.0)):
    """Weighted MSE. Void is 0-5%, so unweighted MSE barely sees void errors; w_void=10
    is a starting point to tune (and an ablation to report)."""
    w = torch.as_tensor(w, device=pred.device, dtype=pred.dtype)
    return ((pred - true) ** 2 * w).mean()


if __name__ == "__main__":
    m = PhaseFractionResNet3D()
    n = sum(p.numel() for p in m.parameters())
    print(f"params: {n/1e6:.2f}M")
    for s in (32, 64):
        print(s, m(torch.randn(2, 1, s, s, s)).shape)
