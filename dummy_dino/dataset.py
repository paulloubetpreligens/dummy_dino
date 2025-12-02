
import torch
from typing import Callable


class DinoDataset(torch.utils.data.Dataset):
    """Dataset wrapper designed for DINO-style self-supervised learning (SSL)"""
    def __init__(
            self,
            base_dataset: torch.utils.data,
            teacher_augmentation: Callable[[torch.Tensor], torch.Tensor], 
            student_augmentation: Callable[[torch.Tensor], torch.Tensor],
            ):
        """
        Initializes the Dataset with the data source and the two augmentation pipelines.

        Args:
            base_dataset: A base PyTorch Dataset.
            teacher_augmentation: A function that takes a Tensor and returns an augmented Tensor for the teacher network.
            student_augmentation: A function that takes a Tensor and returns an augmented Tensor for the student network.
        """
        self.data_source = base_dataset
        self.teacher_augmentation = teacher_augmentation
        self.student_augmentation = student_augmentation


    def __getitem__(self, index):
        X_base = self.data_source[index]
        return self.teacher_augmentation(X_base), self.student_augmentation(X_base)

    def __len__(self):
        return len(self.data_source)