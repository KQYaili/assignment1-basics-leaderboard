import math
import torch

def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    if it < warmup_iters:
        return it/warmup_iters*max_learning_rate
    if it < cosine_cycle_iters:
        return min_learning_rate+0.5*(1+math.cos(math.pi*(it-warmup_iters)/(cosine_cycle_iters-warmup_iters)))*(max_learning_rate-min_learning_rate)
    return min_learning_rate

class adamw(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        assert 0.0 <= lr and 0.0 <= eps and 0.0 <= betas[0] < 1.0 and 0.0 <= betas[1] < 1.0
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super(adamw, self).__init__(params, defaults)

    def step(self, closure = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr, betas,eps,weight_decay = group["lr"], group["betas"], group["eps"],group["weight_decay"]
            for p in group["params"]:
                if p.grad is None: continue
                g = p.grad.data
                state = self.state[p]
                if len(state) == 0:
                    # 第一次更新这个参数时，才按需分配
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)
                state["t"] = state["t"] + 1
                alpha_t=lr*math.sqrt(1-betas[1]**state["t"])/(1-betas[0]**state["t"])
                p.data=p.data-lr*weight_decay*p.data
                state["m"]=betas[0]*state["m"]+(1-betas[0])*g
                state["v"]=betas[1]*state["v"]+(1-betas[1])*g**2
                p.data=p.data-alpha_t*state["m"]/(torch.sqrt(state["v"])+eps)
        return loss