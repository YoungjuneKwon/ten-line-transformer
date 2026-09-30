import sys, math, time, torch, torch.nn as nn
from torch.nn import functional as F

def load_data(path, train_ratio=0.9):
  text = open(path, encoding="utf-8").read()
  chars = sorted(set(text))
  c2i = {c: i for i, c in enumerate(chars)}
  data = torch.tensor([c2i[c] for c in text], dtype=torch.long)
  cut = int(train_ratio * len(data))
  return data[:cut], data[cut:], c2i, {i: c for c, i in c2i.items()}, len(chars)

def get_batch(data, B, T):
  starts = torch.randint(len(data) - T - 1, (B,))
  tokens = torch.stack([data[s:s + T] for s in starts])
  answers = torch.stack([data[s + 1:s + T + 1] for s in starts])
  return tokens, answers

def layer_norm(x, eps=1e-5):
  return (x - x.mean(-1, keepdim=True)) / torch.sqrt(x.var(-1, unbiased=False, keepdim=True) + eps)

def cross_entropy(logits, answers):
  return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), answers.reshape(-1))

class Attention(nn.Module):
  def __init__(self, T, C, H):
    super().__init__()
    self.H = H
    self.hs = C // H
    self.qkv = nn.Linear(C, 3 * H * self.hs, bias=False)
    self.proj = nn.Linear(C, C, bias=False)
    self.register_buffer("mask", torch.tril(torch.ones(T, T)).view(1, 1, T, T))

  def forward(self, x):
    b, t, c = x.shape
    q, k, v = [z.view(b, t, self.H, self.hs).transpose(1, 2)
               for z in self.qkv(x).split(c, dim=2)]
    att = (q @ k.transpose(-2, -1)) / math.sqrt(self.hs)
    att = att.masked_fill(self.mask[:, :, :t, :t] == 0, float("-inf"))
    y = F.softmax(att, dim=-1) @ v
    y = y.transpose(1, 2).contiguous().view(b, t, c)
    return self.proj(y)

class MLP(nn.Module):
  def __init__(self, C):
    super().__init__()
    self.fc = nn.Linear(C, 4 * C, bias=False)
    self.out = nn.Linear(4 * C, C, bias=False)

  def forward(self, x):
    return self.out(F.gelu(self.fc(x)))

class Block(nn.Module):
  def __init__(self, T, C, H):
    super().__init__()
    self.attn = Attention(T, C, H)
    self.mlp = MLP(C)

class Transformer(nn.Module):
  def __init__(self, V, T, C, H, n_block):
    super().__init__()
    self.wte = nn.Parameter(torch.randn(V, C) * 0.02)
    self.wpe = nn.Parameter(torch.randn(T, C) * 0.02)
    self.wue = nn.Parameter(torch.randn(C, V) * 0.02)
    self.blocks = nn.ModuleList([Block(T, C, H) for _ in range(n_block)])

  def forward(self, tokens, answers=None):
    x = self.wte[tokens] + self.wpe[:tokens.shape[1]]
    for blk in self.blocks:
      x = x + blk.attn(layer_norm(x))
      x = x + blk.mlp(layer_norm(x))
    logits = layer_norm(x) @ self.wue
    return logits, None if answers is None else cross_entropy(logits, answers)

def report(model, step, loss, t0, val_batch):
  val_tokens, val_answers = val_batch
  with torch.no_grad():
    _, val_loss = model(val_tokens, val_answers)
  print(f"{step:6d}{loss.item():10.4f}{val_loss.item():10.4f}{time.time() - t0:8.1f}")

def train(model, train_data, val_data, steps, B, T):
  optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4,
                                betas=(0.9, 0.95), weight_decay=0.1)
  print(f"{'step':>6}{'train':>10}{'val':>10}{'sec':>8}")
  report_every = max(1, steps // 10)
  t0 = time.time()
  for step in range(steps + 1):
    tokens, answers = get_batch(train_data, B, T)
    optimizer.zero_grad(set_to_none=True)
    _, loss = model(tokens, answers)
    loss.backward()
    optimizer.step()
    if step % report_every == 0:
      val_batch = get_batch(val_data, B, T)
      report(model, step, loss, t0, val_batch)

def generate(model, n_chars, T, c2i, i2c, seed_text="\n", temperature=0.8):
  seq = torch.tensor([[c2i[c] for c in seed_text]], dtype=torch.long)
  with torch.no_grad():
    for _ in range(n_chars):
      logits, _ = model(seq[:, -T:])
      probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
      seq = torch.cat([seq, torch.multinomial(probs, 1)], dim=1)
  return "".join(i2c[i] for i in seq[0].tolist())

def cli():
  steps = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
  corpus = sys.argv[2] if len(sys.argv) > 2 else "input.txt"
  return steps, corpus

def main(steps, corpus):
  B, T, C, H, n_block = 32, 64, 128, 4, 4
  assert C % H == 0, "C must be divisible by H"
  torch.manual_seed(1337)

  train_data, val_data, c2i, i2c, V = load_data(corpus)
  model = Transformer(V, T, C, H, n_block)
  train(model, train_data, val_data, steps, B, T)
  print("\nsample")
  print(generate(model, 800, T, c2i, i2c))
  return model

if __name__ == "__main__":
  main(*cli())
