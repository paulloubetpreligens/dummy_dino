import torch
from dataclasses import dataclass


@dataclass
class Dino:
    student: torch.nn.Module
    """student model."""
    teacher: torch.nn.Module
    """teacher model."""
    loss_function: torch.nn.Module
    """loss function."""


    @torch.no_grad()
    def update_teacher_weights(self, momentum_rate: float) -> None:
        """
        Update the teacher's weights using an Exponential Moving Average (EMA)
        of the student's weights.
        """
        for teacher_param, student_param in zip(self.teacher.parameters(), self.student.parameters()):
            teacher_param.data.mul_(momentum_rate).add_(student_param.data, alpha=(1 - momentum_rate))

    def train(self, lr:float, data_loader: torch.utils.data.DataLoader, epoch_number: int, optimizer_function: torch.optim.Optimizer)->None:
        for epoch in range(epoch_number):
            total_loss = 0

            optimizer = optimizer_function(params=self.student.parameters(), lr=lr)
            for step, (x_s_batch, x_t_batch) in enumerate(data_loader):
                optimizer.zero_grad()

                out_s = self.student(x_s_batch)
                
                with torch.no_grad():
                    out_t = self.teacher(x_t_batch)
                
                loss = self.loss_function(out_s, out_t)

                loss.backward()
                
                optimizer.step()
   
                self.update_teacher_weights(momentum_rate=0.6)
                
                total_loss += loss.item()