"""Test dino dataset."""

from functools import partial

import torch
import torch.testing

from dummy_dino.core import dataset
from tests.conftest import fake_augmentation


def test_dino_dataset_return_augmented_features(fake_base_dataset):
    """Tests that the dataset returns augmented features."""
    teacher_augmentation = partial(
        fake_augmentation,
        coef=0.9,
    )
    student_augmentation = partial(
        fake_augmentation,
        coef=0.1,
    )
    dino_dataset = dataset.DinoDataset(
        base_dataset=fake_base_dataset,
        teacher_augmentation=teacher_augmentation,
        student_augmentation=student_augmentation,
    )

    assert len(dino_dataset) == 10
    teacher_feature, student_feature = dino_dataset[0]
    torch.testing.assert_close(teacher_feature, torch.ones(10) * 0.9)
    torch.testing.assert_close(student_feature, torch.ones(10) * 0.1)
