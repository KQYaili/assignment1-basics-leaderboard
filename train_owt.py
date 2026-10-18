from cs336_basics.model import TransformerLM,UltimateTransformerLM
from cs336_basics.optimizer import adamw, get_lr_cosine_schedule
from cs336_basics.data import get_batch
from cs336_basics.nn_utils import cross_entropy_loss, clip_gradient
from cs336_basics.serialization import save_checkpoint
import numpy as np
import matplotlib.pyplot as plt
import torch
torch.set_float32_matmul_precision('high')
import time
start_time = time.time()

vocab_size = 32000
context_length = 512
d_model = 512
num_layers = 6
num_heads = 16
d_ff = 1344
rope_theta = 10000.0
device = "cuda"
batch_size = 16
max_iters = 6600
learning_rate = 1.2e-3
warmup_iters = 330
weight_decay = 0.01
grad_clip = 1.0
min_learning_rate = 1e-5
max_time_seconds = 44 * 60

def evaluate(model, valid_memmap, batch_size, context_length, device,it, num_batches=20):
    model.eval()
    with torch.no_grad():
        val_losses_list = []
        for _ in range(num_batches):
            val_x, val_y = get_batch(valid_memmap, batch_size, context_length, device)
            val_losses_list.append(cross_entropy_loss(model(val_x), val_y).item())
        val_loss = sum(val_losses_list) / len(val_losses_list)
    elapsed_mins = (time.time() - start_time) / 60.0
    times_log.append(elapsed_mins)
    val_loss_log.append(val_loss)
    print(f"--- 验证集评估: Step {it} | Time: {elapsed_mins:.1f}m | Val Loss: {val_loss:.4f} ---")
    model.train()
    return val_loss

def draw(times_log, val_loss_log, save_path):
    plt.figure(figsize=(8, 5))
    plt.plot(times_log, val_loss_log, marker='o', label="Validation Loss")
    plt.xlabel("Wall-clock time (minutes)")
    plt.ylabel("Validation Loss")
    plt.title("CS336 OpenWebText Validation Loss")
    plt.grid(True)
    plt.savefig(save_path)
    print("排行榜学习曲线已保存至 data/leaderboard_curve.png！")
    plt.close()

# 1. 实例化模型
model = TransformerLM(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta).to(device)
# model = UltimateTransformerLM(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta).to(device)
# 2. 将 Embedding 权重标准差重置为 0.02（消除 128 倍方差爆炸）
with torch.no_grad():
    torch.nn.init.trunc_normal_(model.token_embeddings.weight, std=0.02, a=-0.06, b=0.06)

# 3. 绑定权重（此时 lm_head 和 embedding 都在正常的尺度上）
model.lm_head.weight = model.token_embeddings.weight
train=np.memmap( "data/owt_train.bin", dtype=np.uint16, mode="r")
valid=np.memmap( "data/owt_valid.bin", dtype=np.uint16, mode="r")
optimizer = adamw(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
times_log = []
val_loss_log = []
for it in range(max_iters):
    lr=get_lr_cosine_schedule(it, learning_rate, min_learning_rate, warmup_iters, max_iters)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    x, y = get_batch(train, batch_size, context_length, device)
    logits = model(x)
    loss = cross_entropy_loss(logits, y)
    # loss.backward()
    (loss/2).backward()
    if it % 2 == 1:
        clip_gradient(model.parameters(), max_l2_norm=grad_clip)
        optimizer.step()
        optimizer.zero_grad()
    elapsed = time.time() - start_time
    if elapsed >= max_time_seconds:
        print(f"已达到 44 分钟时间上限，安全停机，准备最终评测！")
        break
    if it % 100 == 0:
        print(f"it:{it}, lr:{lr}, loss:{loss.item()}")
    if it%1000==0:
        evaluate(model, valid, batch_size, context_length, device, it)
        draw(times_log, val_loss_log, "data/leaderboard_curve.png")
        save_checkpoint(model, optimizer, it, f"data/owt_checkpoint_{it}.pt")
        print(f"已保存 Checkpoint 到 data/owt_checkpoint_{it}.pt")

evaluate(model, valid, batch_size, context_length, device, it)
draw(times_log, val_loss_log, "data/leaderboard_curve.png")
save_checkpoint(model, optimizer, it, f"data/owt_checkpoint_{it}.pt")



