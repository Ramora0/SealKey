# SealKey Architecture Notes

Temporal extension of CorridorKey V2. Residual RGB + direct alpha matting, now with
cross-frame consistency. Target training resolution **480×832**, but inference must
scale up to higher resolutions (≥2K) without retraining, so every decision below is
filtered through "does this still work at 4× the pixels?"

# Key Pitches

Over Corridorkey:
1. Temporal consistency
2. Any color background, blue or green
3. Arbitrary resolution
4. Much wider data
5. No licensing issues

---

## Core Architecture

| Component | Choice |
|-----------|--------|
| Encoder | ConvNeXt-Small, ImageNet pretrained, fully convolutional |
| Decoder | SMP U-Net style, skip connections preserved |
| Input | RGB [B,3,H,W] + hint [B,1,H,W] + prev_alpha [B,1,H,W] |
| Output | RGBA [B,4,H,W] — residual RGB, direct alpha |
| RGB head | `clamp(input + delta, 0, 1)` |
| Alpha head | `sigmoid(logits)` or `hardtanh` — not raw clamp (kills gradients) |
| Temporal state | None — cross-frame info enters only via prev_alpha + hint inputs |
| Normalization | ImageNet for RGB, raw [0,1] for hint and prev_alpha |

**Why no recurrent hidden state:** the model is effectively single-frame per forward
pass; all temporal coupling lives in the prev_alpha channel (and, when supplied, the
hint channel). This keeps the architecture trivially resolution-agnostic, avoids
BPTT, and lets training run on shuffled (frame_t-1, frame_t) pairs instead of long
clips. We accept a temporal-quality ceiling in exchange for simplicity and
resolution robustness; if flicker turns out to need more than prev_alpha
propagation, revisit with window-based attention or a light ConvGRU, not global
attention.

---

## Hint & Prev-Alpha Channels

Two separate single-channel inputs:

- **hint** — external user-provided signal (trimap-ish scribble, box mask, SAM
  click mask, etc.). **Always present**, on every frame including frame 0. The
  model can rely on hint being a real signal; it is never zeroed at inference.
- **prev_alpha** — the detached predicted alpha from t-1. **Zeros on frame 0**
  (no previous frame exists), real prediction thereafter. The model must handle
  the all-zero case gracefully — frame 0 is effectively a single-frame matte
  guided by hint alone.

- prev_alpha is the **only** temporal mechanism — there is no hidden state, so
  everything the model knows about the past is whatever fits in this one channel.
- Optional upgrade: flow-warp prev_alpha before feeding it in, using a light flow
  net or RAFT-small. Skip on v1.
- The 4th input channel (hint) and 5th (prev_alpha) are randomly initialized in
  the first conv — not free pretrained signal. Fine, but don't expect ImageNet
  transfer on those slices.

---

## Resolution-Agnostic Design Rules

Training at 480×832 but deploying higher means **nothing in the model may bake in
a fixed spatial size.**

- No learned positional embeddings.
- No fixed-size pooling heads (no global avg pool feeding an MLP, etc.).
- If attention is ever added, it must be window-based (Swin) or neighborhood
  attention. No global attention.
- ConvNeXt is fully convolutional out of the box. Don't break that.

---

## Training Setup

### Clip length
- No recurrent state → no BPTT. Training operates on **frame pairs**
  `(frame_{t-1}, frame_t)` with prev_alpha supplied from the GT (or from a
  detached forward pass on frame_{t-1}, scheduled-sampling style, to close the
  train/inference gap).
- Longer sequences are only needed at eval time, to measure drift over many
  frames. Short pairs are enough to train the temporal losses below.

### Multi-scale training
- Random scale augmentation within 0.75×–1.5× of 480×832.
- This is the primary defense against receptive-field mismatch at inference scale.

### Validation
- Hold out a few **high-res** clips. Run them periodically during training.
- Edge quality and temporal stability at high-res is where fully-conv models
  surprise you badly. Catch it early.

### Loss
Three loss terms:
- BCE or L1 on alpha
- L1 on RGB weighted by GT_alpha * (1 - GT_alpha) to only focus on edges
- Small L1 reg on all deltas


---

## Loss Design

The architecture is table stakes — losses do the actual work. All temporal losses
must be **scale-normalized** so they behave identically at training and inference
resolutions.

### Per-frame losses
- Alpha L1 + L2
- Laplacian / gradient loss on alpha (edge sharpness)
- Composited-RGB loss: `composite(pred_rgba, bg) vs composite(gt_rgba, bg)` over
  multiple random backgrounds. This is what makes unmixing actually learn.

### Temporal losses
- **Flow-warped alpha consistency:**
  `L1(alpha_t, warp(alpha_{t-1}, flow_{t-1→t})) * (1 - occlusion_mask)`
  Use an off-the-shelf flow model (RAFT) as a frozen teacher.
- **Static-region temporal gradient:** on regions with near-zero flow, penalize
  `|alpha_t - alpha_{t-1}|`. Kills flicker on static backgrounds.
- **Composited-RGB temporal consistency:** catches unmixing flicker that alpha-only
  losses miss.

### Scale normalization gotcha
Flow magnitude is in pixels. 10 px/frame at 480×832 becomes 40 px/frame at
1920×3328. Normalize flow to frame-fraction or [-1,1] before any loss uses it, or
inference-time temporal behavior will not match training.

---

## Composite Convention

Decide up front and document it in the dataloader, loss, and any export path:

- **Premultiplied** (`rgb * alpha` is what's stored): residual `rgb + delta` is
  modeling premultiplied color. Loss must compare premultiplied.
- **Unpremultiplied** (full foreground color, alpha separate): residual is modeling
  unpremultiplied. Compositing downstream applies `rgb * alpha + bg * (1 - alpha)`.

Mixing these is the #1 subtle bug in matting pipelines. Pick one, assert it in the
dataset, and write the composited-RGB loss to match.

---

## Out of Scope (for now)

- Memory-attention banks (XMem/SAM2 style). Revisit only if occlusion / re-ID
  becomes a real failure mode, and only with window-based attention.
- 3D or (2+1)D convolutions. prev_alpha feedback covers the useful part at lower
  cost; revisit only if temporal quality plateaus.
- ConvGRU / recurrent hidden state. Removed — see Core Architecture rationale.
- Learned flow. Use a frozen off-the-shelf flow model for losses; don't co-train.

---

## Expected Gains vs CorridorKey V2 (single-frame)

- Flicker reduction: large and visible. Main user-facing win.
- Edge stability through motion blur: moderate.
- Occlusion handling: minimal — that's what memory-attention solves, and we're
  explicitly not building that.
- Per-frame alpha quality on static frames: roughly neutral. Temporal model should
  not regress the single-frame baseline; if it does, the prev_alpha channel is
  being over-trusted and dragging the per-frame prediction toward a stale mask.
