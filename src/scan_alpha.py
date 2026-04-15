"""Quickly scan generated RGBA clips for alpha leakage.

Each clip is either a .zip of RGBA PNG frames (what wan_alpha_comfyui writes)
or a directory of RGBA PNG frames (already-extracted). A random frame is shown
composited over a high-frequency, high-contrast background so any unintended
transparency in the middle of the subject is immediately obvious. From the
viewer you can keep, throw out, or re-encode the clip with internal alpha
holes filled in.

Keys:
    n / space / right  next clip
    p / left           previous clip
    r                  resample a new random frame
    s                  cycle background style
    f                  toggle preview of the hole-fill on the current frame
    k                  keep, advance
    t                  throw out (move to rejected/), advance
    a                  apply fix to the whole clip (preview first, then confirm)
    q / esc            quit
"""

from __future__ import annotations

import argparse
import io
import random
import shutil
import zipfile
from pathlib import Path

import cv2
import numpy as np


def list_clips(root: Path) -> list[Path]:
    """Return sorted list of clips under `root` — either .zip files or dirs of PNGs."""
    out: list[Path] = []
    for p in root.iterdir():
        if "rejected" in p.parts or "fixed" in p.parts:
            continue
        if p.is_file() and p.suffix.lower() == ".zip":
            out.append(p)
        elif p.is_dir():
            if any(c.suffix.lower() == ".png" for c in p.iterdir()):
                out.append(p)
    return sorted(out)


def _frame_names(clip: Path) -> list[str]:
    if clip.suffix.lower() == ".zip":
        with zipfile.ZipFile(clip) as zf:
            return sorted(n for n in zf.namelist() if n.lower().endswith(".png"))
    return sorted(p.name for p in clip.iterdir() if p.suffix.lower() == ".png")


def _read_png(clip: Path, name: str) -> np.ndarray:
    if clip.suffix.lower() == ".zip":
        with zipfile.ZipFile(clip) as zf:
            data = zf.read(name)
        arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    else:
        arr = cv2.imread(str(clip / name), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise RuntimeError(f"failed to read PNG {name} from {clip}")
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGRA)
    elif arr.shape[2] == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2BGRA)
    return cv2.cvtColor(arr, cv2.COLOR_BGRA2RGBA)


def load_random_frame(clip: Path) -> np.ndarray:
    names = _frame_names(clip)
    if not names:
        raise RuntimeError(f"no PNG frames in {clip}")
    return _read_png(clip, random.choice(names))


def load_all_frames(clip: Path) -> tuple[list[str], list[np.ndarray]]:
    names = _frame_names(clip)
    return names, [_read_png(clip, n) for n in names]


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
    mask[1:-1, 1:-1] = 1 - floodable
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


def render(rgba: np.ndarray, style: int, clip: Path, idx: int, total: int,
           status: str, preview_fix: bool) -> np.ndarray:
    shown = fix_rgba(rgba) if preview_fix else rgba
    h, w = shown.shape[:2]
    bg = make_background(h, w, style)
    comp = composite(shown, bg)
    alpha_vis = np.repeat(shown[..., 3:4], 3, axis=-1)
    panel = np.concatenate([comp, alpha_vis], axis=1)
    panel_bgr = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
    return annotate(panel_bgr, [
        f"[{idx + 1}/{total}] {clip.name}",
        f"status: {status}   bg-style: {style % 3}   preview-fix: {'on' if preview_fix else 'off'}",
        "n=next p=prev r=resample s=bg f=preview-fix k=keep t=throw a=apply-fix q=quit",
    ])


def render_confirm(rgba: np.ndarray, style: int, clip: Path) -> np.ndarray:
    fixed = fix_rgba(rgba)
    h, w = rgba.shape[:2]
    bg = make_background(h, w, style)
    left = composite(rgba, bg)
    right = composite(fixed, bg)
    gap = np.full((h, 8, 3), 255, dtype=np.uint8)
    panel = np.concatenate([left, gap, right], axis=1)
    panel_bgr = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
    cv2.putText(panel_bgr, "ORIGINAL", (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(panel_bgr, "ORIGINAL", (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(panel_bgr, "FIXED", (w + 18, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(panel_bgr, "FIXED", (w + 18, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    return annotate(panel_bgr, [
        f"CONFIRM FIX — {clip.name}",
        "y / enter = apply to whole clip   any other key = cancel",
        "r = resample a different frame first",
    ])


def write_fixed_clip(clip: Path, names: list[str], frames: list[np.ndarray], out_root: Path) -> Path:
    """Write the fixed frames back in the same format as the source (zip or dir)."""
    out_root.mkdir(parents=True, exist_ok=True)
    if clip.suffix.lower() == ".zip":
        dest = out_root / clip.name
        with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for name, rgba in zip(names, frames):
                bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
                ok, buf = cv2.imencode(".png", bgra)
                if not ok:
                    raise RuntimeError(f"PNG encode failed for {name}")
                zf.writestr(name, buf.tobytes())
        return dest
    dest = out_root / clip.name
    dest.mkdir(parents=True, exist_ok=True)
    for name, rgba in zip(names, frames):
        bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
        cv2.imwrite(str(dest / name), bgra)
    return dest


def main() -> None:
    default_dir = Path("/fs/scratch/PAS2836/lees_stuff/sealkey_wan_alpha")
    ap = argparse.ArgumentParser()
    ap.add_argument("directory", type=Path, nargs="?", default=default_dir,
                    help=f"Folder of clips to scan (default: {default_dir})")
    ap.add_argument("--results", type=Path, default=Path("scan_results.txt"))
    ap.add_argument("--rejected-dir", type=Path, default=None,
                    help="Where to move thrown-out clips (default: <directory>/rejected)")
    ap.add_argument("--fixed-dir", type=Path, default=None,
                    help="Where to write hole-filled clips (default: <directory>/fixed)")
    ap.add_argument("--max-width", type=int, default=1200)
    args = ap.parse_args()

    rejected_dir = args.rejected_dir or (args.directory / "rejected")
    fixed_dir = args.fixed_dir or (args.directory / "fixed")

    clips = list_clips(args.directory)
    if not clips:
        raise SystemExit(f"no clips (.zip or PNG dirs) found in {args.directory}")

    verdicts: dict[str, str] = {}
    if args.results.exists():
        for line in args.results.read_text().splitlines():
            if "\t" in line:
                v, p = line.split("\t", 1)
                verdicts[p] = v

    idx = 0
    style = 0
    preview = False
    rgba = load_random_frame(clips[idx])

    win = "alpha-scan"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def show() -> None:
        clip = clips[idx]
        img = render(rgba, style, clip, idx, len(clips),
                     verdicts.get(str(clip), "unrated"), preview)
        if img.shape[1] > args.max_width:
            scale = args.max_width / img.shape[1]
            img = cv2.resize(img, (args.max_width, int(img.shape[0] * scale)),
                             interpolation=cv2.INTER_AREA)
        cv2.imshow(win, img)

    def advance() -> None:
        nonlocal idx, rgba, preview
        idx = (idx + 1) % len(clips)
        preview = False
        rgba = load_random_frame(clips[idx])

    show()
    while True:
        key = cv2.waitKey(0) & 0xFF
        clip = clips[idx]

        if key in (ord("q"), 27):
            break
        elif key in (ord("n"), ord(" "), 83):
            advance()
        elif key in (ord("p"), 81):
            idx = (idx - 1) % len(clips)
            preview = False
            rgba = load_random_frame(clips[idx])
        elif key == ord("r"):
            rgba = load_random_frame(clip)
        elif key == ord("s"):
            style += 1
        elif key == ord("f"):
            preview = not preview
        elif key == ord("k"):
            verdicts[str(clip)] = "keep"
            advance()
        elif key == ord("t"):
            rejected_dir.mkdir(parents=True, exist_ok=True)
            dest = rejected_dir / clip.name
            shutil.move(str(clip), str(dest))
            verdicts[str(clip)] = f"thrown -> {dest}"
            clips.pop(idx)
            if not clips:
                break
            idx %= len(clips)
            preview = False
            rgba = load_random_frame(clips[idx])
        elif key == ord("a"):
            confirmed = False
            while True:
                cimg = render_confirm(rgba, style, clip)
                if cimg.shape[1] > args.max_width:
                    scale = args.max_width / cimg.shape[1]
                    cimg = cv2.resize(cimg, (args.max_width, int(cimg.shape[0] * scale)),
                                      interpolation=cv2.INTER_AREA)
                cv2.imshow(win, cimg)
                ckey = cv2.waitKey(0) & 0xFF
                if ckey == ord("r"):
                    rgba = load_random_frame(clip)
                    continue
                confirmed = ckey in (ord("y"), 13, 10)
                break
            if not confirmed:
                print("[fix] cancelled")
            else:
                print(f"[fix] loading {clip.name} ...")
                names, frames = load_all_frames(clip)
                print(f"[fix] filling holes across {len(frames)} frames ...")
                fixed = [fix_rgba(f) for f in frames]
                out_path = write_fixed_clip(clip, names, fixed, fixed_dir)
                print(f"[fix] wrote -> {out_path}")
                verdicts[str(clip)] = f"fixed -> {out_path}"
                advance()

        if not clips:
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
