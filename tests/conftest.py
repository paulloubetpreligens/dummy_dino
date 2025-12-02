"""Configuration file for pytest, containing shared fixtures and helper classes."""
import torch
import pytest

from functools import partial
from dummy_dino.dataset import DinoDataset


class FakeBaseDataset(torch.utils.data.Dataset):
    """A fake basedataset that returns features of ones."""

    def __init__(self, num_samples: int, feature_dim: int):
        self.num_samples = num_samples
        self.features = torch.ones(num_samples, feature_dim)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx) -> torch.Tensor:
        return self.features[idx]

@pytest.fixture
def fake_base_dataset()->FakeBaseDataset:
    return FakeBaseDataset(num_samples=10, feature_dim=10)


def fake_augmentation(input: torch.Tensor, coef: float)->torch.Tensor:
    return input * coef


@pytest.fixture
def fake_dino_dataset(fake_base_dataset:FakeBaseDataset)->DinoDataset:
    teacher_augmentation = partial(
        fake_augmentation, 
        coef=0.9,
    )
    student_augmentation = partial(
        fake_augmentation, 
        coef=0.1,
    )
    return DinoDataset(
        base_dataset=fake_base_dataset,
        teacher_augmentation=teacher_augmentation,
        student_augmentation=student_augmentation,
    )