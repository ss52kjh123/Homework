# Python packages
from termcolor import colored
from typing import Dict
import copy

# PyTorch & Pytorch Lightning
from lightning.pytorch import LightningModule
from lightning.pytorch.loggers.wandb import WandbLogger
from torch import nn
from torchvision import models
from torchvision.models.alexnet import AlexNet
import torch
from thop import profile

# Custom packages
from src.metric import MyAccuracy, MyF1Score
import src.config as cfg
from src.util import show_setting


# [TODO: Optional] Rewrite this class if you want
class MyNetwork(AlexNet):
    """
    * ImageNet pre-trained AlexNet을 가져와 feature extractor 일부를
      BatchNorm + Dropout이 들어간 형태로 살짝 바꾸고,
    * classifier 마지막 FC를 (num_classes) 로 교체합니다.
    * 필요하면 freeze_until 파라미터로 앞쪽 layer 동결도 가능.
    """
    def __init__(self,
                 num_classes: int = 200,
                 pretrained: bool = True,
                 freeze_until: int = 0            # feature layer 몇 개까지 freeze?
                 ):
        # ───────────────────── base AlexNet 로드 ─────────────────────
        if pretrained:
            super().__init__(weights=AlexNet_Weights.IMAGENET1K_V1)
        else:
            super().__init__()

        # ───────────────────── feature extractor 수정 ─────────────────────
        # (예시) 첫 Conv stride를 2→1로 바꾸고 BatchNorm 추가
        # ① conv1 stride 변경
        self.features[0] = nn.Conv2d(
            in_channels=3, out_channels=64,
            kernel_size=11, stride=1, padding=2
        )
        # ② conv1 다음 위치에 BatchNorm·ReLU·Dropout 살짝 추가
        #    (원본은 이미 ReLU가 있음 → 새 블록 삽입)
        new_block = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25)
        )
        # features 는 nn.Sequential 이라 insert 불가 → 새 리스트로 재구성
        self.features = nn.Sequential(
            *([self.features[0]] +               # 수정된 conv1
              list(new_block) +                  # 추가 블록
              list(self.features.children())[1:] # 기존 conv2~conv5 등
             )
        )

        # ───────────────────── 마지막 FC 교체 ─────────────────────
        in_features = self.classifier[6].in_features   # 4096 그대로
        self.classifier[6] = nn.Linear(in_features, num_classes)

        # ─────────────────────(선택) 일부 layer freeze ─────────────────────
        if freeze_until > 0:
            for idx, (_, p) in enumerate(self.features.named_parameters()):
                if idx < freeze_until:
                    p.requires_grad = False


    # forward는 원본과 동일하니 건드릴 필요 없음
    # (참고: AlexNet.forward() 내부 구현과 동일하게 유지)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class SimpleClassifier(LightningModule):
    def __init__(self,
                 model_name: str = 'resnet18',
                 num_classes: int = 200,
                 optimizer_params: Dict = dict(),
                 scheduler_params: Dict = dict(),
        ):
        super().__init__()
        self.save_hyperparameters()
        

        # Network
        if model_name == 'MyNetwork':
            self.model = MyNetwork()
        else:
            models_list = models.list_models()
            assert model_name in models_list, f'Unknown model name: {model_name}. Choose one from {", ".join(models_list)}'
            self.model = models.get_model(model_name, num_classes=num_classes)

        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()

        # Metric
        self.accuracy = MyAccuracy()
        self.f1score = MyF1Score(num_classes=num_classes)
        # Hyperparameters
        self.save_hyperparameters()

    def on_train_start(self):
        show_setting(cfg)

    def on_fit_start(self):
        # dummy 입력을 넣어서 FLOPs와 Params 계산
        dummy_input = torch.randn(1, 3, 64, 64).to(self.device)
        flops, params = profile(self.model, inputs=(dummy_input,), verbose=False)
        
        

    def configure_optimizers(self):
        optim_params = copy.deepcopy(self.hparams.optimizer_params)
        optim_type = optim_params.pop('type')
        optimizer = getattr(torch.optim, optim_type)(self.parameters(), **optim_params)

        scheduler_params = copy.deepcopy(self.hparams.scheduler_params)
        scheduler_type = scheduler_params.pop('type')
        scheduler = getattr(torch.optim.lr_scheduler, scheduler_type)(optimizer, **scheduler_params)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch)
        accuracy = self.accuracy(scores, y)
        self.log_dict({'loss/train': loss, 'accuracy/train': accuracy},
                      on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch)
        accuracy = self.accuracy(scores, y)
        self.f1score.update(scores, y)
        self.log_dict({'loss/val': loss, 'accuracy/val': accuracy},
                      on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self._wandb_log_image(batch, batch_idx, scores, frequency = cfg.WANDB_IMG_LOG_FREQ)

    def on_validation_epoch_end(self):
        # F1 계산 결과를 log
        f1 = self.f1score.compute().mean()  # 평균 F1
        self.log("f1/val", f1, prog_bar=True, logger=True)
        self.f1score.reset()
        if self.current_epoch == 0 and isinstance(self.logger, WandbLogger):
            dummy_input = torch.randn(1, 3, 64, 64).to(self.device)
            flops, params = profile(self.model, inputs=(dummy_input,), verbose=False)
            self.logger.experiment.log({
                "model/params": params,
                "model/flops": flops
            }, step=self.global_step)

    def _common_step(self, batch):
        x, y = batch
        scores = self.forward(x)
        loss = self.loss_fn(scores, y)
        return loss, scores, y

    def _wandb_log_image(self, batch, batch_idx, preds, frequency = 100):
        if not isinstance(self.logger, WandbLogger):
            if batch_idx == 0:
                self.print(colored("Please use WandbLogger to log images.", color='blue', attrs=('bold',)))
            return

        if batch_idx % frequency == 0:
            x, y = batch
            preds = torch.argmax(preds, dim=1)
            self.logger.log_image(
                key=f'pred/val/batch{batch_idx:5d}_sample_0',
                images=[x[0].to('cpu')],
                caption=[f'GT: {y[0].item()}, Pred: {preds[0].item()}'])