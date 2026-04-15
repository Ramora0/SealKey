"""Quickly scan generated videos for alpha leakage.

Shows a random frame composited over a high-frequency, high-contrast
background so any unintended transparency in the middle of the subject
is immediately obvious. From the viewer you can keep, throw out, or
re-encode the video with internal alpha holes filled in.

Keys:
    n / space / right  next video
    p / left           previous video
    r                  resample a new random frame
    s                  cycle background style
    f                  toggle preview of the hole-fill on the current frame
    k                  keep, advance
    t                  throw out (move to rejected/), advance
    a                  apply fix to the whole video, save to fixed/, advance
    q / esc            quit
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import av
import cv2
import numpy as np

VIDEO_EXTS = {".mp4", ".webm", ".mkv", ".mov", ".avi", ".gif"}


def list_videos(root: Path) -> list[Path]:
    out = []
    for p in root.rglob("*"):
        if p.suffix.lower() not in VIDEO_EXTS:
            continue
        # skip our own outputs
        if "rejected" in p.parts or "fixed" in p.parts:
            continue
        out.append(p)
    return sorted(out)


def decode_frame(path: Path, index: int | None) -> np.ndarray:
    """Decode a single frame as HxWx4 uint8 RGBA. If index is None, picks random."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        n = stream.frames or 0
        target = index if index is not None else (random.randint(0, n - 1) if n > 0 else None)
        chosen = None
        for i, frame in enumerate(container.decode(stream)):
            chosen = frame
            if target is not None and i >= target:
                break
        if chosen is None:
            raise RuntimeError(f"no frames in {path}")
        try:
            return chosen.to_ndarray(format="rgba")
        except Exception:
            rgb = chosen.to_ndarray(format="rgb24")
            alpha = np.full(rgb.shape[:2] + (1,), 255, dtype=np.uint8)
            return np.concatenate([rgb, alpha], axis=-1)


def decode_all(path: Path) -> tuple[list[np.ndarray], float]:
    frames: list[np.ndarray] = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        fps = float(stream.average_rate) if stream.average_rate else 24.0
        for frame in container.decode(stream):
            try:
                arr = frame.to_ndarray(format="rgba")
            except Exception:
                rgb = frame.to_ndarray(format="rgb24")
                alpha = np.full(rgb.shape[:2] + (1,), 255, dtype=np.uint8)
                arr = np.concatenate([rgb, alpha], axis=-1)
            frames.append(arr)
    if not frames:
        raise RuntimeError(f"no frames in {path}")
    return frames, fps


def fill_internal_holes(alpha: np.ndarray, flood_thresh: int = 200, edge_dilate: int = 6) -> np.ndarray:
    """Set interior sub-opaque pixels to 255, leaving the natural edge alone.

    Flood from the image border through every pixel with alpha < flood_thresh.
    The outer soft edge is a connected low-alpha band anchored at the truly-
    transparent background, so the flood sweeps it. Then dilate the flooded
    region by `edge_dilate` pixels to give the edge a safety margin. Any sub-
    flood_thresh pixel outside that margin is an interior hole → set to 255.
    """
    h, w = alpha.shape
    floodable = (alpha < flood_thresh).astype(np.uint8)
    mask = np.zeros((h + 2, w + 2), np.uint8)
    mask[1:-1, 1:-1] = 1 - floodable  # pixels >= flood_thresh block the flood
    canvas = floodable.copy()
    for x in range(w):
        if canvas[0, x] and mask[1, x + 1] == 0:
            cv2.floodFill(canvas, mask, (x, 0), 2, flags=4)
        if canvas[h - 1, x] and mask[h, x + 1] == 0:
            cv2.floodFill(canvas, mask, (x, h - 1), 2, flags=4)
    for y in range(h):
        if canvas[y, 0] and mask[y + 1, 1] == 0:
            cv2.floodFill(canvas, mask, (0, y), 2, flags=4)
        if canvas[y, w - 1] and mask[y + 1, w] == 0:
            cv2.floodFill(canvas, mask, (w - 1, y), 2, flags=4)
    protected = (canvas == 2).astype(np.uint8)
    if edge_dilate > 0:
        k = 2 * edge_dilate + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        protected = cv2.dilate(protected, kernel)
    holes = (~protected.astype(bool)) & (alpha < 255)
    out = alpha.copy()
    out[holes] = 255
    return out


def fix_rgba(rgba: np.ndarray) -> np.ndarray:
    fixed = rgba.copy()
    fixed[..., 3] = fill_internal_holes(rgba[..., 3])
    return fixed


def make_background(h: int, w: int, style: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    if style % 3 == 0:
        yy, xx = np.indices((h, w))
        m = ((xx // 4) + (yy // 4)) % 2
        return np.where(m[..., None], np.array([255, 0, 255], np.uint8),
                        np.array([0, 255, 255], np.uint8)).astype(np.uint8)
    if style % 3 == 1:
        n = rng.integers(0, 2, size=(h, w), dtype=np.uint8) * 255
        return np.stack([n, 255 - n, n], axis=-1)
    yy, xx = np.indices((h, w))
    m = (xx // 2) % 2
    return np.where(m[..., None], np.uint8(255), np.uint8(0)).repeat(3, axis=-1)


def composite(rgba: np.ndarray, bg: np.ndarray) -> np.ndarray:
    a = rgba[..., 3:4].astype(np.float32) / 255.0
    return np.clip(rgba[..., :3].astype(np.float32) * a + bg.astype(np.float32) * (1 - a),
                   0, 255).astype(np.uint8)


def annotate(img: np.ndarray, lines: list[str]) -> np.ndarray:
    out = img.copy()
    y = 24
    for line in lines:
        cv2.putText(out, line, (10, y + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        y += 22
    return out


def render(rgba: np.ndarray, style: int, path: Path, idx: int, total: int,
           status: str, preview_fix: bool) -> np.ndarray:
    shown = fix_rgba(rgba) if preview_fix else rgba
    h, w = shown.shape[:2]
    bg = make_background(h, w, style)
    comp = composite(shown, bg)
    alpha_vis = np.repeat(shown[..., 3:4], 3, axis=-1)
    panel = np.concatenate([comp, alpha_vis], axis=1)
    panel_bgr = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
    return annotate(panel_bgr, [
        f"[{idx + 1}/{total}] {path.name}",
        f"status: {status}   bg-style: {style % 3}   preview-fix: {'on' if preview_fix else 'off'}",
        "n=next p=prev r=resample s=bg f=preview-fix k=keep t=throw a=apply-fix q=quit",
    ])


def encode_rgba_webm(frames: list[np.ndarray], fps: float, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    container = av.open(str(out_path), mode="w")
    try:
        stream = container.add_stream("libvpx-vp9", rate=int(round(fps)))
        stream.width = w
        stream.height = h
        stream.pix_fmt = "yuva420p"
        stream.options = {"crf": "18", "b:v": "0", "auto-alt-ref": "0"}
        for arr in frames:
            frame = av.VideoFrame.from_ndarray(arr, format="rgba")
            frame = frame.reformat(format="yuva420p")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("directory", type=Path)
    ap.add_argument("--results", type=Path, default=Path("scan_results.txt"))
    ap.add_argument("--rejected-dir", type=Path, default=None,
                    help="Where to move thrown-out videos (default: <directory>/rejected)")
    ap.add_argument("--fixed-dir", type=Path, default=None,
                    help="Where to write hole-filled videos (default: <directory>/fixed)")
    ap.add_argument("--max-width", type=int, default=1200)
    args = ap.parse_args()

    rejected_dir = args.rejected_dir or (args.directory / "rejected")
    fixed_dir = args.fixed_dir or (args.directory / "fixed")

    videos = list_videos(args.directory)
    if not videos:
        raise SystemExit(f"no videos found under {args.directory}")

    verdicts: dict[str, str] = {}
    if args.results.exists():
        for line in args.results.read_text().splitlines():
            if "\t" in line:
                v, p = line.split("\t", 1)
                verdicts[p] = v

    idx = 0
    style = 0
    preview = False
    rgba = decode_frame(videos[idx], None)

    win = "alpha-scan"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def show() -> None:
        path = videos[idx]
        img = render(rgba, style, path, idx, len(videos),
                     verdicts.get(str(path), "unrated"), preview)
        if img.shape[1] > args.max_width:
            scale = args.max_width / img.shape[1]
            img = cv2.resize(img, (args.max_width, int(img.shape[0] * scale)),
                             interpolation=cv2.INTER_AREA)
        cv2.imshow(win, img)

    def advance() -> None:
        nonlocal idx, rgba, preview
        idx = (idx + 1) % len(videos)
        preview = False
        rgba = decode_frame(videos[idx], None)

    show()
    while True:
        key = cv2.waitKey(0) & 0xFF
        path = videos[idx]

        if key in (ord("q"), 27):
            break
        elif key in (ord("n"), ord(" "), 83):
            advance()
        elif key in (ord("p"), 81):
            idx = (idx - 1) % len(videos)
            preview = False
            rgba = decode_frame(videos[idx], None)
        elif key == ord("r"):
            rgba = decode_frame(path, None)
        elif key == ord("s"):
            style += 1
        elif key == ord("f"):
            preview = not preview
        elif key == ord("k"):
            verdicts[str(path)] = "keep"
            advance()
        elif key == ord("t"):
            rejected_dir.mkdir(parents=True, exist_ok=True)
            dest = rejected_dir / path.name
            shutil.move(str(path), str(dest))
            verdicts[str(path)] = f"thrown -> {dest}"
            videos.pop(idx)
            if not videos:
                break
            idx %= len(videos)
            preview = False
            rgba = decode_frame(videos[idx], None)
        elif key == ord("a"):
            print(f"[fix] decoding {path.name} ...")
            frames, fps = decode_all(path)
            print(f"[fix] filling holes across {len(frames)} frames ...")
            fixed = [fix_rgba(f) for f in frames]
            out_path = fixed_dir / (path.stem + ".webm")
            print(f"[fix] encoding -> {out_path}")
            encode_rgba_webm(fixed, fps, out_path)
            verdicts[str(path)] = f"fixed -> {out_path}"
            advance()

        if not videos:
            break
        show()

    cv2.destroyAllWindows()
    if verdicts:
        args.results.write_text("\n".join(f"{v}\t{p}" for p, v in verdicts.items()) + "\n")
        n_keep = sum(1 for v in verdicts.values() if v == "keep")
        n_throw = sum(1 for v in verdicts.values() if v.startswith("thrown"))
        n_fix = sum(1 for v in verdicts.values() if v.startswith("fixed"))
        print(f"saved {len(verdicts)} verdicts  keep={n_keep} thrown={n_throw} fixed={n_fix}")


if __name__ == "__main__":
    main()
