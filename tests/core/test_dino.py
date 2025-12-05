import pytest
import torch

from dummy_dino.core.dataset import DinoDataset
from dummy_dino.core.dino import Dino


@pytest.fixture
def student_model() -> torch.nn.Module:
    """Fixture for the student model."""
    return torch.nn.Linear(10, 10)


@pytest.fixture
def teacher_model() -> torch.nn.Module:
    """Fixture for the teacher model."""
    return torch.nn.Linear(10, 10)


@pytest.fixture
def dino_model(student_model: torch.nn.Module, teacher_model: torch.nn.Module) -> Dino:
    """Fixture for the DINO model instance."""
    return Dino(
        student=student_model, teacher=teacher_model, loss_function=torch.nn.MSELoss()
    )


def test_update_teacher_weights(dino_model: Dino):
    """
    Tests if the teacher model weights are correctly updated using EMA
    from the student model weights.
    """
    # 1. Given
    torch.nn.init.ones_(dino_model.student.weight)
    torch.nn.init.zeros_(dino_model.teacher.weight)

    # 2. When
    dino_model.update_teacher_weights(momentum_rate=0.5)

    # 3. Then
    expected_weight = torch.ones(10, 10) * 0.5
    assert torch.allclose(
        dino_model.teacher.weight, expected_weight
    ), "Teacher weights were not updated correctly."


class FakeDinoDataset(torch.utils.data.Dataset):
    """A fake dataset that returns a single random feature vector.

    TODO: here we could create a specific interface for the dinon dataset with augment for studetn and teacher.
    """

    def __init__(self, num_samples: 64, feature_dim: 10):
        self.num_samples = num_samples
        self.features = torch.randn(num_samples, feature_dim)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx]


def test_train(dino_model: Dino, fake_dino_dataset: DinoDataset):
    # Given

    initial_s_weight = dino_model.student.weight.clone()
    initial_t_weight = dino_model.teacher.weight.clone()

    # When
    dino_model.train(
        lr=0.5,
        data_loader=fake_dino_dataset,
        epoch_number=1,
        optimizer_function=torch.optim.SGD,
    )

    # Then

    # Student weights should have changed
    assert not torch.allclose(
        dino_model.student.weight, initial_s_weight
    ), "Student weights did not update."

    # Teacher weights should have changed
    assert not torch.allclose(
        dino_model.teacher.weight, initial_t_weight
    ), "Teacher weights did not update."
