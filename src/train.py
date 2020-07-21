import logging.config
import os
from datetime import datetime
from logging import getLogger
from typing import Any, Dict, List, Tuple

import albumentations as albu
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import yaml

import utils
from datasets import InstanceSegmentationDataset
from models import get_instance_segmentation_model
from trainer import InstanceSegmentationTrainer
from utils import collate_fn


def create_experiment_directories() -> str:
    # Setup directory that saves the experiment results.
    dirname: str = datetime.now().strftime('%Y%m%d_%H-%M-%S')
    save_dir: str = os.path.join('../experiments', dirname)
    os.makedirs(save_dir, exist_ok=False)
    weights_dir: str = os.path.join(save_dir, 'weights')
    os.makedirs(weights_dir, exist_ok=False)
    return save_dir


if __name__ == '__main__':
    utils.seed_everything(seed=428)
    sns.set()
    save_dir: str = create_experiment_directories()

    with open('logger_conf.yaml', 'r') as f:
        log_config: Dict[str, Any] = yaml.safe_load(f.read())
        logging.config.dictConfig(log_config)

    logger = getLogger(__name__)
    cfg_dict: Dict[str, Any] = utils.load_yaml('config.yml')
    cfg: utils.DotDict = utils.DotDict(cfg_dict)

    dtrain = InstanceSegmentationDataset(
        albu.core.serialization.from_dict(cfg.albumentations.train.todict())
    )
    dvalid = InstanceSegmentationDataset(
        albu.core.serialization.from_dict(cfg.albumentations.eval.todict())
    )

    indices = list(range(len(dtrain)))
    dtrain = torch.utils.data.Subset(dtrain, indices[:-50])  # type: ignore
    dvalid = torch.utils.data.Subset(dvalid, indices[-50:])  # type: ignore

    train_loader = torch.utils.data.DataLoader(
        dtrain, batch_size=cfg.batch_size, shuffle=True,
        collate_fn=collate_fn, drop_last=True
    )

    valid_loader = torch.utils.data.DataLoader(
        dvalid, batch_size=1, shuffle=False,
        collate_fn=collate_fn
    )

    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = get_instance_segmentation_model(
        cfg.num_classes, cfg.pretrained
    )
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=cfg.optimizer.lr,
        momentum=cfg.optimizer.momentum,
        weight_decay=cfg.optimizer.weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=3, gamma=0.1
    )

    trainer = InstanceSegmentationTrainer(model, optimizer)
    train_losses: Dict[str, List[float]] = {
        'loss_classifier': [], 'loss_box_reg': [], 'loss_mask': [],
        'loss_objectness': [], 'loss_rpn_box_reg': [], 'loss_sum': []
    }
    valid_losses: Dict[str, List[float]] = {
        'loss_classifier': [], 'loss_box_reg': [], 'loss_mask': [],
        'loss_objectness': [], 'loss_rpn_box_reg': [], 'loss_sum': []
    }
    for epoch in range(cfg.num_epochs):
        train_loss: Dict[str, float] = trainer.epoch_train(train_loader)
        lr_scheduler.step()
        valid_loss: Dict[str, float] = trainer.epoch_eval(valid_loader)

        for k, v in train_loss.items():
            train_losses[k].append(v)
        train_losses['loss_sum'].append(sum([v for v in train_loss.values()]))
        for k, v in valid_loss.items():
            valid_losses[k].append(v)
        valid_losses['loss_sum'].append(sum([v for v in valid_loss.values()]))

        logger.info(f'EPOCH: [{epoch + 1}/{cfg.num_epochs}]')
        logger.info('TRAIN_LOSS:')
        logger.info(train_loss)
        logger.info('VALID_LOSS')
        logger.info(valid_loss)

    weights_path: str = os.path.join(save_dir, 'weights', 'weights.pth')
    torch.save(model.state_dict(), weights_path)

    utils.dump_yaml(os.path.join(save_dir, 'config.yml'), cfg.todict())
    # Plot metrics
    keys: Tuple[str, ...] = (
        'loss_classifier', 'loss_box_reg', 'loss_mask', 'loss_objectness',
        'loss_rpn_box_reg', 'loss_sum'
    )
    plt.tight_layout()
    plt.figure(figsize=(12, 8))

    for idx, key in enumerate(keys):
        plt.subplot(2, 3, idx + 1)
        plt.plot(train_losses[key], label='train')
        plt.plot(valid_losses[key], label='valid')
        plt.title(key)
        plt.legend()
    plt.savefig(os.path.join(save_dir, 'fig.png'))
    plt.clf()
