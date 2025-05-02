from torchmetrics import Metric
import torch

# ────────────────────────────────
# 1) F1 Score (per-class, one-vs-rest)
# ────────────────────────────────
class MyF1Score(Metric):
    """
    • num_classes: 전체 클래스 개수
    • update()    : confusion-matrix 누적
    • compute()   : 클래스별 F1 (tensor[num_classes]) 반환
    """
    def __init__(self, num_classes: int):
        super().__init__(dist_sync_on_step=False)
        self.num_classes = num_classes
        self.add_state(
            "confmat",
            default=torch.zeros(num_classes, num_classes, dtype=torch.int64),
            dist_reduce_fx="sum"
        )

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # preds: (B, C) → (B,) 로 argmax
        if preds.ndim == 2:
            preds = torch.argmax(preds, dim=1)

        preds  = preds.view(-1).long()
        target = target.view(-1).long()

        if preds.shape != target.shape:
            raise ValueError(f"Shape mismatch: preds{preds.shape} vs target{target.shape}")

        # confusion matrix 갱신
        for p, t in zip(preds, target):
            self.confmat[p, t] += 1

    def compute(self) -> torch.Tensor:
        cm = self.confmat.float()                       # [P, T]
        f1_list = []

        for c in range(self.num_classes):
            TP = cm[c, c]
            FP = cm[c, :].sum() - TP
            FN = cm[:, c].sum() - TP

            precision = TP / (TP + FP + 1e-8)
            recall    = TP / (TP + FN + 1e-8)
            f1        = 2 * precision * recall / (precision + recall + 1e-8)
            f1_list.append(f1)

        return torch.tensor(f1_list)   # shape: [num_classes]


# ────────────────────────────────
# 2) Accuracy
# ────────────────────────────────
class MyAccuracy(Metric):
    """
    간단한 Top-1 정확도
    """
    def __init__(self):
        super().__init__(dist_sync_on_step=False)
        self.add_state('total',   default=torch.tensor(0, dtype=torch.int64), dist_reduce_fx='sum')
        self.add_state('correct', default=torch.tensor(0, dtype=torch.int64), dist_reduce_fx='sum')

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # 1) argmax
        if preds.ndim == 2:
            preds = torch.argmax(preds, dim=1)

        preds  = preds.view(-1)
        target = target.view(-1)

        # 2) shape 확인
        if preds.shape != target.shape:
            raise ValueError(f"Shape mismatch: preds{preds.shape} vs target{target.shape}")

        # 3) 올바른 예측 수
        correct = torch.sum(preds == target)

        # 4) 누적
        self.correct += correct
        self.total   += target.numel()

    def compute(self) -> torch.Tensor:
        return self.correct.float() / self.total.float()