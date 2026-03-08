import numpy as np
import os
import re
from collections import Counter

# ---------- data ----------
in_path = os.path.join(os.getcwd(), "MaschineInteligence", "book_clean.txt")
text = open(in_path, "r", encoding="utf-8").read()

# Tokenizer: words + punctuation + newlines
# - "\w+" catches letter sequences (unicode-friendly)
# - punctuation as separate tokens
# - newline becomes special token <NL>
raw_tokens = re.findall(r"\n|\w+|[^\w\s]", text, flags=re.UNICODE)
tokens = ["<NL>" if t == "\n" else t for t in raw_tokens]

# Vocab cap (important for RAM). Tune this.
max_vocab = 10000
special = ["<UNK>", "<NL>"]

counts = Counter(tokens)
# keep most common, but ensure <NL> exists (it will, but let's be explicit)
most_common = [w for (w, _) in counts.most_common(max_vocab - len(special))]
# remove specials if they appear in text (rare), we'll add them manually
most_common = [w for w in most_common if w not in special]
vocab = special + most_common[: max_vocab - len(special)]

word_to_ix = {w: i for i, w in enumerate(vocab)}
ix_to_word = {i: w for w, i in word_to_ix.items()}
V = len(vocab)

# map tokens to indices (OOV -> <UNK>)
UNK = word_to_ix["<UNK>"]
data = [word_to_ix.get(t, UNK) for t in tokens]

print("tokens:", len(tokens), "vocab:", V)

# ---------- hyperparams ----------
H = 128          # hidden size
seq_len = 30     # tokens per unroll (word-level: smaller than char-level is often fine)
lr = 0.1

# ---------- params ----------
rng = np.random.default_rng(0)
# Wx acts like an embedding matrix: Wx[:, ix] is the word embedding for token ix
Wx = rng.normal(0, 0.01, size=(H, V))   # token -> hidden (embedding)
Wh = rng.normal(0, 0.01, size=(H, H))   # hidden -> hidden
b  = np.zeros((H, 1))
Wy = rng.normal(0, 0.01, size=(V, H))   # hidden -> logits over vocab
by = np.zeros((V, 1))

# Adagrad memory
mWx = np.zeros_like(Wx); mWh = np.zeros_like(Wh); mWy = np.zeros_like(Wy)
mb  = np.zeros_like(b);  mby = np.zeros_like(by)

def softmax(z):
    z = z - np.max(z)      # stability
    e = np.exp(z)
    return e / np.sum(e)

def loss_and_grads(inputs, targets, hprev):
    """
    inputs, targets: lists of token indices (len=T)
    hprev: (H,1)
    """
    T = len(inputs)
    hs = [hprev] + [None] * T
    ps = [None] * T
    loss = 0.0

    # ---- forward ----
    for t in range(T):
        ix = inputs[t]
        # embedding lookup instead of building one-hot:
        xcol = Wx[:, ix:ix+1]                   # (H,1)
        hs[t+1] = np.tanh(xcol + Wh @ hs[t] + b)
        logits = Wy @ hs[t+1] + by              # (V,1)
        ps[t] = softmax(logits)
        loss += -np.log(ps[t][targets[t], 0] + 1e-12)

    # ---- backward (BPTT) ----
    dWx = np.zeros_like(Wx); dWh = np.zeros_like(Wh); dWy = np.zeros_like(Wy)
    db  = np.zeros_like(b);  dby = np.zeros_like(by)
    dh_next = np.zeros_like(hprev)

    for t in reversed(range(T)):
        dy = ps[t].copy()
        dy[targets[t]] -= 1.0                   # softmax+CE gradient wrt logits

        dWy += dy @ hs[t+1].T
        dby += dy

        dh = Wy.T @ dy + dh_next
        dh_raw = (1 - hs[t+1] * hs[t+1]) * dh   # tanh'

        db += dh_raw
        dWh += dh_raw @ hs[t].T

        ix = inputs[t]
        dWx[:, ix:ix+1] += dh_raw               # sparse update into embedding column

        dh_next = Wh.T @ dh_raw

    # clip (avoid exploding gradients)
    for g in (dWx, dWh, dWy, db, dby):
        np.clip(g, -5, 5, out=g)

    return loss, (dWx, dWh, dWy, db, dby), hs[-1]

def detokenize(tok_list):
    # cheap detokenizer: spaces between words, no space before punctuation
    out = []
    for t in tok_list:
        if t == "<NL>":
            out.append("\n")
        elif len(out) == 0:
            out.append(t)
        elif re.match(r"^[^\w\s]$", t):         # punctuation token
            out.append(t)
        else:
            # if previous is newline, don't add leading space
            if out[-1].endswith("\n"):
                out.append(t)
            else:
                out.append(" " + t)
    return "".join(out)

def sample(h, seed_ix, n=80):
    ix = seed_ix
    out_tokens = []
    for _ in range(n):
        xcol = Wx[:, ix:ix+1]
        h = np.tanh(xcol + Wh @ h + b)
        p = softmax(Wy @ h + by)
        ix = int(rng.choice(V, p=p.ravel()))
        out_tokens.append(ix_to_word[ix])
    return detokenize(out_tokens)

# ---------- training loop ----------
hprev = np.zeros((H, 1))
p = 0
smooth_loss = -np.log(1.0 / V) * seq_len

for it in range(1, 5001):
    if p + seq_len + 1 >= len(data):
        hprev = np.zeros((H, 1))
        p = 0

    inputs  = data[p : p + seq_len]
    targets = data[p + 1 : p + seq_len + 1]

    loss, (dWx, dWh, dWy, db, dby), hprev = loss_and_grads(inputs, targets, hprev)
    smooth_loss = 0.999 * smooth_loss + 0.001 * loss

    # Adagrad update
    for param, dparam, mem in [
        (Wx, dWx, mWx), (Wh, dWh, mWh), (Wy, dWy, mWy), (b, db, mb), (by, dby, mby)
    ]:
        mem += dparam * dparam
        param += -lr * dparam / (np.sqrt(mem) + 1e-8)

    if it % 500 == 0:
        print(f"iter {it}, loss {smooth_loss:.3f}")
        print(sample(hprev, seed_ix=inputs[0], n=80))
        print("-"*60)

    p += seq_len
