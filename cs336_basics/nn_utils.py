import torch
import math

def SiLU(x):
    return x*torch.sigmoid(x)

def softmax(v: torch.Tensor,i) -> torch.Tensor:
    c=torch.max(v,dim=i,keepdim=True).values
    exp=torch.exp(v-c)
    return exp/torch.sum(exp,dim=i,keepdim=True)

def cross_entropy_loss(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    c=torch.max(inputs,dim=-1,keepdim=True).values
    target_logits = torch.gather(inputs, dim=-1, index=targets.unsqueeze(-1))
    log_sum_exp = c + torch.log(torch.sum(torch.exp(inputs - c), dim=-1, keepdim=True))
    loss = log_sum_exp - target_logits
    return loss.mean()

def scaled_dot_product_attention(Q,K,V,mask=None) -> torch.Tensor:
    d_k=Q.shape[-1]
    # (batch_size, ..., seq_len, d_k) @ (batch_size, ..., d_v,seq_len) -> (batch_size, ..., seq_len,seq_len)
    attention_scores=Q@K.transpose(-1,-2)/ math.sqrt(d_k)
    if mask is not None:
        attention_scores=attention_scores.masked_fill(~mask,-torch.inf)
    return softmax(attention_scores,-1)@V

def clip_gradient(parameters, max_l2_norm,eps=1e-6):
    total_sum=0
    l=[]
    for p in parameters:
        if p.grad is not None:
            l.append(p.grad)
    for i in range(len(l)):
        total_sum+=(l[i].data**2).sum()
    g_2=math.sqrt(total_sum)
    if g_2<max_l2_norm:
        return
    scale=max_l2_norm/(g_2+eps)
    for i in range (len(l)):
        l[i].data*=scale

