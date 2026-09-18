"""
Model architecture for the BPSK / QPSK / GMSK modulation classifier.

This MUST match the architecture used during training on Kaggle exactly —
load_state_dict() only restores weights, not structure, so any mismatch in
layer sizes/order will either throw an error or silently load garbage.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModClassifier(nn.Module):
    def __init__(self, num_classes=3, input_len=1024):
        super().__init__()
        self.conv1 = nn.Conv1d(2, 32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool = nn.MaxPool1d(2)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # x expected shape: (batch, 2, 1024) — channels-first IQ
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.global_pool(x).squeeze(-1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)