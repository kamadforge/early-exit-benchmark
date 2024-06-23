import loss_landscapes as ll
from loss_landscapes import metrics
import torch


class ListLoss(metrics.Metric):
    """Computes a specified loss function over specified input-output pairs."""

    def __init__(self, loss_fn, inputs: torch.Tensor, target: torch.Tensor):
        super().__init__()
        self.loss_fn = loss_fn
        self.inputs = inputs
        self.target = target

    def __call__(self, model_wrapper: ll.ModelWrapper) -> float:
        return [
            self.loss_fn(out, self.target).item()
            for out in model_wrapper.forward(self.inputs)
        ]


class DataloaderListLoss(metrics.Metric):
    """Computes a specified loss function over dataloader"""

    def __init__(
        self,
        loss_fn,
        loader: torch.utils.data.DataLoader,
        n_heads: int,
        gen_device: torch.device,
    ):
        super().__init__()
        self.loss_fn = loss_fn
        self.loader = loader
        self.n_heads = n_heads
        self.gen_device = gen_device

    def __call__(self, model_wrapper: ll.ModelWrapper) -> float:
        outs, labels = self._evaluate_with_dataloader(model_wrapper)
        return [self.loss_fn(out, labels).item() for out in outs]

    def _evaluate_with_dataloader(
        self,
        model_wrapper: ll.ModelWrapper,
    ):
        model_wrapper.eval()
        batch_outputs = []
        batch_labels = []
        with torch.inference_mode():
            with self.gen_device:
                for X, y in self.loader:
                    output = model_wrapper.forward(X)
                    y_preds = [y_pred.detach() for y_pred in output]
                    batch_outputs.append(y_preds)
                    batch_labels.append(y)
        head_preds = []
        labels = torch.cat(batch_labels)
        for i in range(self.n_heads):
            head_outputs = torch.cat(
                [batch_output[i] for batch_output in batch_outputs]
            )
            head_preds.append(head_outputs)
        return head_preds, labels
