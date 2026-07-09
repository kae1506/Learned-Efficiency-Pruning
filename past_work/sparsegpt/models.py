"""
Self-contained copies of the CIFAR conv-net architectures used in the research,
so a checkpoint (experiments/checkpoints/cifar_{cnn,mid,big}.pt) can be rebuilt
here without importing from scripts/. Kept byte-for-byte compatible with
scripts/base/train_cifar.py so the saved state_dicts load cleanly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2470, 0.2435, 0.2616)


class CIFARNet(nn.Module):
    """LeNet-style CNN (~63K params). Prunable target: fc1, fc2 (fc3 = classifier)."""
    def __init__(self, output_dim: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool  = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16 * 5 * 5, 128)
        self.fc2   = nn.Linear(128, 64)
        self.fc3   = nn.Linear(64, output_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class CIFARNetMid(nn.Module):
    """Mid conv-net (~1.22M). Prunable target: fc1, fc2, fc3 (fc4 = classifier)."""
    def __init__(self, output_dim: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3,   32,  3, padding=1); self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32,  64,  3, padding=1); self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128,  3, padding=1); self.bn3 = nn.BatchNorm2d(128)
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(128 * 4 * 4, 512)
        self.fc2   = nn.Linear(512, 128)
        self.fc3   = nn.Linear(128, 64)
        self.fc4   = nn.Linear(64, output_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


class CIFARNetBig(nn.Module):
    """Big conv-net (~10.4M). Prunable target: fc1, fc2, fc3 (fc4 = classifier)."""
    def __init__(self, output_dim: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1);    self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 256, 3, padding=1);  self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 512, 3, padding=1); self.bn3 = nn.BatchNorm2d(512)
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(512 * 4 * 4, 1024)
        self.fc2   = nn.Linear(1024, 512)
        self.fc3   = nn.Linear(512, 256)
        self.fc4   = nn.Linear(256, output_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


# arch string (as stored in the checkpoint) -> (class, ordered prunable FC-head names)
# The classifier (last fc) is deliberately EXCLUDED, matching the research, which
# prunes hidden neurons of the MLP head and never the output layer.
ARCHS = {
    "lenet": (CIFARNet,    ["fc1", "fc2"]),
    "cnn":   (CIFARNet,    ["fc1", "fc2"]),      # cifar_cnn.pt stores arch="lenet"; alias
    "mid":   (CIFARNetMid, ["fc1", "fc2", "fc3"]),
    "big":   (CIFARNetBig, ["fc1", "fc2", "fc3"]),
}


def load_checkpoint(path: str, device):
    """Rebuild the model from a training checkpoint {'config': {...}, 'state_dict'}."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    arch = cfg.get("arch", "lenet")
    key = "lenet" if arch in ("lenet", "cnn") else arch
    cls, prunable = ARCHS[key]
    model = cls(cfg["output_dim"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, prunable
