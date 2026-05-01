from manifolds.sphere import Sphere
import torch
import math
from torch.optim import Optimizer


class PolarEmbeddingsOptimizer(Optimizer):

    """Optimizer wrapper that applies manifold-aware updates for polar embeddings."""
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0):
        """Initialize the PolarEmbeddingsOptimizer object and its experiment state."""
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError(
                "Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(PolarEmbeddingsOptimizer, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Apply one optimizer update step."""
        if closure is not None:
            with torch.enable_grad():
                closure()

        for group in self.param_groups:

            mode = group.get('polar_mode', 'euclidean')
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError(
                        'PolarEmbeddingsOptimizer does not support sparse gradients')

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(
                        p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(
                        p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(
                    grad, grad, value=1 - beta2)

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                step_size = group['lr'] * \
                    math.sqrt(bias_correction2) / bias_correction1

                p.addcdiv_(exp_avg, denom, value=-step_size)

                if mode == 'lat':
                    p.clamp_(min=0.0, max=math.pi)
                elif mode == 'long':
                    p.remainder_(2*math.pi)


class RiemannianAdam(Optimizer):

    """Adam optimizer variant with post-step spherical parameter projection."""
    def __init__(self, params, lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        """Initialize the RiemannianAdam object and its experiment state."""
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.manifold = Sphere()
        super(RiemannianAdam, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Apply one optimizer update step."""
        loss = None

        if closure is not None:
            with torch.no_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError(
                        "Riemannian Adam does not support sparse gradients yet.")

                state = self.state[p]

                is_spherical = getattr(p, 'is_manifold', False)

                if is_spherical:
                    if len(state) == 0:
                        state['step'] = 0

                        state['exp_avg'] = torch.zeros_like(p)
                        state['exp_avg_sq'] = torch.zeros_like(p)
                    state['step'] += 1

                    beta1, beta2 = group['betas']

                    grad_tan = self.manifold.proj_tan(p, grad)

                    if group['weight_decay'] != 0:
                        grad_tan = grad_tan.add(p, alpha=group['weight_decay'])
                    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                    exp_avg_transported = self.manifold.parallel_transport(
                        p, p, exp_avg)

                    exp_avg.copy_(beta1*exp_avg_transported+(1-beta1)*grad_tan)
                    grad_norm_sq = torch.sum(
                        grad_tan*grad_tan, dim=-1, keepdim=True)

                    exp_avg_sq.copy_(beta2*exp_avg_sq+(1-beta2)*grad_norm_sq)

                    bias_correction1 = 1-beta1**state['step']
                    bias_correction2 = 1-beta2**state['step']

                    m_hat = exp_avg/bias_correction1
                    v_hat = exp_avg_sq/bias_correction2

                    direction = m_hat/(torch.sqrt(v_hat)+group['eps'])
                    step_size = -group['lr']
                    p_new = self.manifold.expmap(p, step_size*direction)

                    p.copy_(p_new)
                else:
                    if len(state) == 0:
                        state['step'] = 0
                        state['exp_avg'] = torch.zeros_like(p)
                        state['exp_avg_sq'] = torch.zeros_like(p)

                    state['step'] += 1
                    beta1, beta2 = group['betas']

                    if group['weight_decay'] != 0:
                        grad = grad.add(p, alpha=group['weight_decay'])

                    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                    exp_avg.mul_(beta1).add_(grad, alpha=1-beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1-beta2)

                    bias_correction1 = 1-beta1**state['step']
                    bias_correction2 = 1-beta2**state['step']

                    denom = (exp_avg_sq.sqrt() /
                             math.sqrt(bias_correction2)).add_(group['eps'])
                    p.addcdiv_(exp_avg, denom, value=-
                               group['lr'] / bias_correction1)

                    p.copy_(p)
