import torch
import torch.nn.functional as F


class GradientLogger:

    """Collect gradient norms for selected model layers during training."""
    def __init__(self):
        """Initialize the GradientLogger object and its experiment state."""
        self.gradients = {}

    def log_gradients_for_layers(self, loss_component, named_parameters_to_log, retain_graph=True):
        """Backpropagate a loss and record gradient norms for selected layers."""
        param_names = [name for name, param in named_parameters_to_log]
        params = [param for name, param in named_parameters_to_log]

        grads = torch.autograd.grad(
            outputs=loss_component,
            inputs=params,
            grad_outputs=torch.ones_like(loss_component),
            retain_graph=True,
            allow_unused=True
        )

        self.gradients = {}
        for name, grad in zip(param_names, grads):
            if grad is not None:
                self.gradients[f"grads_taxo/{name}"] = {
                    "l2_norm": grad.norm(2).item(),
                    "mean_abs": grad.abs().mean().item(),
                    "std": grad.std().item()
                }

    def get_loggable_dict(self):

        """Return collected gradient statistics as plain Python values."""
        log_dict = {}
        for name, stats in self.gradients.items():
            for stat_name, value in stats.items():
                log_dict[f"{name}/{stat_name}"] = value
        return log_dict

    def clear(self):
        """Clear all stored gradient statistics."""
        self.gradients = {}
