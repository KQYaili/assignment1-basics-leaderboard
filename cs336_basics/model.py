from torch import nn
import torch
import math

from cs336_basics.nn_utils import SiLU, scaled_dot_product_attention


class linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super(linear, self).__init__()
        self.weight=nn.Parameter(nn.init.trunc_normal_(torch.Tensor(out_features,in_features),
                                                  std=math.sqrt(2/(in_features+out_features)),
                                                  a=-3*math.sqrt(2/(in_features+out_features)),
                                                  b=3*math.sqrt(2/(in_features+out_features))
                                                    ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y=x@self.weight.T
        return y

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super(Embedding, self).__init__()
        self.weight=nn.Parameter(nn.init.trunc_normal_(torch.Tensor(num_embeddings,embedding_dim),a=-3,b=3))

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.weight=nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        d_model=x.shape[-1]
        result = x/(torch.sqrt(torch.sum(x**2,dim=-1,keepdim=True)/d_model+self.eps))*self.weight
        # Return the result in the original dtype
        return result.to(in_dtype)

class SwiGLU(nn.Module):
    def __init__(self,d_model: int , d_ff: int, device=None, dtype=None):
        super(SwiGLU, self).__init__()
        self.w1= linear(d_model,d_ff,device,dtype)
        self.w2= linear(d_ff,d_model,device,dtype)
        self.w3 = linear(d_model,d_ff, device, dtype)

    def forward(self,x) -> torch.Tensor:
        return self.w2(SiLU(self.w1(x))*self.w3(x))

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super(RotaryPositionalEmbedding, self).__init__()
        inv_freq=theta**(-torch.arange(0,d_k,2)/d_k)
        net=torch.outer(torch.arange(0,max_seq_len),inv_freq)
        net=net.repeat_interleave(2,dim=-1)
        self.register_buffer('cos',net.cos(), persistent=False)
        self.register_buffer('sin',net.sin(), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        pos_cos=self.cos[token_positions]
        pos_sin=self.sin[token_positions]
        r1=x[..., ::2]
        r2=x[..., 1::2]
        r_new=torch.stack([-r2,r1],dim=-1).flatten(-2,-1)
        return x*pos_cos+r_new*pos_sin

class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, theta=None,max_seq_len=None,device=None):
        super(MultiheadSelfAttention, self).__init__()
        self.d_model=d_model
        self.num_heads=num_heads
        assert d_model % num_heads == 0
        self.d_q=d_model // num_heads
        self.d_k=d_model // num_heads
        self.d_v=d_model // num_heads
        self.q_proj=linear(d_model,d_model)
        self.k_proj=linear(d_model,d_model)
        self.v_proj=linear(d_model,d_model)
        self.output_proj=linear(d_model,d_model)
        self.r=None
        if theta is not None and max_seq_len is not None:
            self.r=RotaryPositionalEmbedding(theta,d_model // num_heads,max_seq_len,device)

    def forward(self, x: torch.Tensor, token_positions = None) -> torch.Tensor:
        q=self.q_proj(x)
        k=self.k_proj(x)
        v=self.v_proj(x)
        query=q.view(q.shape[0],q.shape[1],self.num_heads,self.d_q).transpose(1,2)
        key=k.view(k.shape[0],k.shape[1],self.num_heads,self.d_k).transpose(1,2)
        value=v.view(v.shape[0],v.shape[1],self.num_heads,self.d_v).transpose(1,2)
        if self.r is not None:
            if token_positions is None:
                token_positions=torch.arange(x.shape[1],device=x.device)
            if token_positions.ndim == 2:
                token_positions=token_positions.unsqueeze(1)
            query=self.r(query,token_positions)
            key=self.r(key,token_positions)
        mask=torch.tril(torch.ones([x.shape[1],x.shape[1]],dtype=torch.bool,device=x.device))
        attention_scores=scaled_dot_product_attention(query, key, value, mask)
        attention_scores=attention_scores.transpose(1,2).reshape(x.shape[0],x.shape[1],x.shape[2])
        return self.output_proj(attention_scores)

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len=None, theta=None, device=None):
        super(TransformerBlock, self).__init__()
        self.ln1 = RMSNorm(d_model)

        self.attn = MultiheadSelfAttention(d_model,num_heads,theta,max_seq_len,device)

        self.ln2 = RMSNorm(d_model)

        self.ffn = SwiGLU(d_model,d_ff)

    def forward(self, x: torch.Tensor,token_positions = None) -> torch.Tensor:
        y=x+self.attn(self.ln1(x),token_positions)
        out=y+self.ffn(self.ln2(y))
        return out

class TransformerLM(nn.Module):
    def __init__(self, vocab_size :int,context_length :int,d_model :int,num_layers:int,num_heads :int,d_ff:int,rope_theta :float,device=None):
        super(TransformerLM, self).__init__()
        self.token_embeddings=Embedding(vocab_size,d_model)
        self.layers = nn.ModuleList([TransformerBlock(d_model,num_heads,d_ff,context_length,rope_theta,device) for _ in range(num_layers)])
        self.ln_final=RMSNorm(d_model)
        self.lm_head=linear(d_model,vocab_size)

    def forward(self, in_indices: torch.Tensor) -> torch.Tensor:
        tokens=self.token_embeddings(in_indices)
        for layer in self.layers:
            tokens=layer(tokens)
        tokens=self.ln_final(tokens)
        return self.lm_head(tokens)

class MultiheadSelfAttention_QK(nn.Module):
    def __init__(self, d_model: int, num_heads: int, theta=None,max_seq_len=None,device=None):
        super(MultiheadSelfAttention_QK, self).__init__()
        self.d_model=d_model
        self.num_heads=num_heads
        assert d_model % num_heads == 0
        self.d_q=d_model // num_heads
        self.d_k=d_model // num_heads
        self.d_v=d_model // num_heads
        self.q_norm = RMSNorm(self.d_k)
        self.k_norm = RMSNorm(self.d_k)
        self.q_proj=linear(d_model,d_model)
        self.k_proj=linear(d_model,d_model)
        self.v_proj=linear(d_model,d_model)
        self.output_proj=linear(d_model,d_model)
        self.r=None
        if theta is not None and max_seq_len is not None:
            self.r=RotaryPositionalEmbedding(theta,d_model // num_heads,max_seq_len,device)

    def forward(self, x: torch.Tensor, token_positions = None) -> torch.Tensor:
        q=self.q_proj(x)
        k=self.k_proj(x)
        v=self.v_proj(x)
        query=q.view(q.shape[0],q.shape[1],self.num_heads,self.d_q).transpose(1,2)
        key=k.view(k.shape[0],k.shape[1],self.num_heads,self.d_k).transpose(1,2)
        value=v.view(v.shape[0],v.shape[1],self.num_heads,self.d_v).transpose(1,2)
        query = self.q_norm(query)
        key = self.k_norm(key)
        if self.r is not None:
            if token_positions is None:
                token_positions=torch.arange(x.shape[1],device=x.device)
            if token_positions.ndim == 2:
                token_positions=token_positions.unsqueeze(1)
            query=self.r(query,token_positions)
            key=self.r(key,token_positions)
        mask=torch.tril(torch.ones([x.shape[1],x.shape[1]],dtype=torch.bool,device=x.device))
        attention_scores=scaled_dot_product_attention(query, key, value, mask)
        attention_scores=attention_scores.transpose(1,2).reshape(x.shape[0],x.shape[1],x.shape[2])
        return self.output_proj(attention_scores)

class AttentionResidualOp(nn.Module):
    def __init__(self, d_model: int):
        super(AttentionResidualOp, self).__init__()
        self.w = nn.Parameter(torch.zeros(d_model))
        self.norm = RMSNorm(d_model)
    def forward(self, sources) -> torch.Tensor:
        V = torch.stack(sources, dim=0)
        K = self.norm(V)
        logits = K@self.w
        alpha=logits.softmax(dim=0)
        h=(alpha.unsqueeze(-1)*V).sum(dim=0)
        return h

class UltimateTransformerLM(nn.Module):
    def __init__(self, vocab_size :int,context_length :int,d_model :int,num_layers:int,num_heads :int,d_ff:int,rope_theta :float,device=None):
        super(UltimateTransformerLM, self).__init__()
        self.token_embeddings=Embedding(vocab_size,d_model)
        # 生成 Stage 1（4 个独立的 Q-K Block）
        self.stage_1 = nn.ModuleList()
        for _ in range(num_layers // 2):
            block = TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta, device)
            block.attn = MultiheadSelfAttention_QK(d_model, num_heads, rope_theta, context_length, device)
            self.stage_1.append(block)
        # 生成 Stage 2（4 个独立的 Q-K Block）
        self.stage_2 = nn.ModuleList()
        for _ in range(num_layers // 2):
            block = TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta, device)
            block.attn = MultiheadSelfAttention_QK(d_model, num_heads, rope_theta, context_length, device)
            self.stage_2.append(block)
        self.attn_res_1=AttentionResidualOp(d_model)
        # self.attn_res_2=AttentionResidualOp(d_model)
        self.ln_final=RMSNorm(d_model)
        self.lm_head=linear(d_model,vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        history=[]
        h_0=self.token_embeddings(x)
        history.append(h_0)
        h=h_0
        for stage in self.stage_1:
            h=stage(h)
        h_1=h
        history.append(h_1)
        h_bridge=self.attn_res_1(history)
        h=h_bridge
        for stage in self.stage_2:
            h=stage(h)
        h_2=h
        # history.append(h_2)
        # h_final=self.attn_res_2(history)
        return self.lm_head(self.ln_final(h_2))





