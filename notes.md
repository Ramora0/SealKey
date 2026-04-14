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
4. Trained on a wide variety of poses

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
| Temporal state | ConvGRU cells at decoder scales 1/16, 1/8, 1/4 |
| Hidden dims | 32–64 ch per GRU (RVM showed this is enough) |
| Normalization | ImageNet for RGB, raw [0,1] for hint and prev_alpha |

**Why recurrence, not memory attention:** ConvGRU scales linearly with pixel count.
Cross-attention over spatial tokens scales quadratically — cheap at 480×832
(~1.5k tokens at 1/16), brutal at 2K+ (~30k tokens). Designing around attention now
means ripping it out later. Accept a small quality ceiling in exchange for
resolution robustness.

---

## Hint & Prev-Alpha Channels

Hint is user-provided on frame 0 only. For t>0, feed the **detached, predicted
alpha from t-1** as the "hint" channel (or as a separate prev_alpha channel — cleaner
to keep both, with hint zeroed after frame 0).

- This is the highest-leverage temporal mechanism. Most perceived "temporal
  intelligence" comes from prev_alpha propagation, not from the GRU state.
- Optional upgrade: flow-warp prev_alpha before feeding it in, using a light flow
  net or RAFT-small. Skip on v1.
- The 4th input channel (hint) and 5th (prev_alpha) are randomly initialized in the
  first conv — not free pretrained signal. Fine, but don't expect ImageNet transfer
  on those slices.

---

## Resolution-Agnostic Design Rules

Training at 480×832 but deploying higher means **nothing in the model may bake in
a fixed spatial size.**

- No learned positional embeddings.
- No fixed-size pooling heads (no global avg pool feeding an MLP, etc.).
- ConvGRU hidden state shape = feature map shape → auto-scales. Keep it that way.
- If attention is ever added, it must be window-based (Swin) or neighborhood
  attention. No global attention.
- ConvNeXt is fully convolutional out of the box. Don't break that.

---

## Training Setup

### Clip length and BPTT
- 8–16 frame clips at 480×832 fit comfortably.
- Truncated BPTT: detach hidden state every 4–8 frames during training to bound
  memory.
- Longer clips matter — temporal losses need enough frames to see drift.

### Multi-scale training
- Random scale augmentation within 0.75×–1.5× of 480×832.
- This is the primary defense against receptive-field mismatch at inference scale.
- Without it, ConvGRU hidden states trained at a single scale tend to under-smooth
  at high-res because the temporal receptive field in world-space shrinks.

### Validation
- Hold out a few **high-res** clips. Run them periodically during training.
- Edge quality and temporal stability at high-res is where fully-conv models
  surprise you badly. Catch it early.

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
- 3D or (2+1)D convolutions. ConvGRU subsumes the useful part at lower cost.
- Learned flow. Use a frozen off-the-shelf flow model for losses; don't co-train.
- Multi-hint UX (hints on later frames). prev_alpha feedback covers the common case.

---

## Expected Gains vs CorridorKey V2 (single-frame)

- Flicker reduction: large and visible. Main user-facing win.
- Edge stability through motion blur: moderate.
- Occlusion handling: minimal — that's what memory-attention solves, and we're
  explicitly not building that.
- Per-frame alpha quality on static frames: roughly neutral. Temporal model should
  not regress the single-frame baseline; if it does, the GRU is overfitting to
  training clip statistics.
