from typing import Dict

import torch
import torch.nn as nn


class InstanceSegmentationTrainer:

    def __init__(self, model: nn.Module, optimizer):
        self._model: nn.Module = model
        self.optimizer = optimizer
        self.device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    def epoch_train(self, train_loader) -> Dict[str, float]:
        self._model.train()
        losses: Dict[str, float] = {
            'loss_classifier': 0.,
            'loss_box_reg': 0.,
            'loss_mask': 0.,
            'loss_objectness': 0.,
            'loss_rpn_box_reg': 0.
        }
        num_samples: int = 0
        for images, targets in train_loader:
            images = list(image.to(self.device) for image in images)
            targets = [
                {k: v.to(self.device) for k, v in t.items()} for t in targets
            ]
            loss_dict: Dict[str, torch.Tensor] = self._model(images, targets)
            for k, v in loss_dict.items():
                losses[k] += v.item()

            loss = sum(loss for loss in loss_dict.values())

            self.optimizer.zero_grad()
            loss.backward()  # type: ignore
            self.optimizer.step()

            num_samples += len(images)

        return {k: v / num_samples for k, v in losses.items()}

    def epoch_eval(self, eval_loader) -> Dict[str, float]:
        self._model.train()
        losses: Dict[str, float] = {
            'loss_classifier': 0.,
            'loss_box_reg': 0.,
            'loss_mask': 0.,
            'loss_objectness': 0.,
            'loss_rpn_box_reg': 0.
        }
        num_samples: int = 0
        for images, targets in eval_loader:
            images = list(image.to(self.device) for image in images)
            targets = [
                {k: v.to(self.device) for k, v in t.items()} for t in targets
            ]
            loss_dict: Dict[str, torch.Tensor] = self._model(images, targets)
            for k, v in loss_dict.items():
                losses[k] += v.item()

            num_samples += len(images)
        return {k: v / num_samples for k, v in losses.items()}
