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


def fill_internal_holes(alpha: np.ndarray, flood_thresh: int = 200,
                        edge_dilate: int = 6, keep_pct: int = 20,
                        near_zero: int = 10) -> np.ndarray:
    """Fill noisy interior holes while preserving real intentional ones.

    1. Flood from the image border through every pixel with alpha < flood_thresh
       (sweeps the natural soft edge since it's anchored at true background).
    2. Dilate the flooded region by `edge_dilate` px as a safety margin.
    3. Whatever remains outside that margin and under flood_thresh is a
       candidate hole. Split into connected components.
    4. For each component, compute the % of its pixels with alpha < near_zero.
         - % >= keep_pct   -> real hole (subject is genuinely transparent here), skip.
         - % <  keep_pct   -> semi-transparent noise, bump to 255.
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

    candidates = ((canvas == 1) & (~protected.astype(bool))).astype(np.uint8)
    num, labels = cv2.connectedComponents(candidates, connectivity=4)
    out = alpha.copy()
    for comp in range(1, num):
        comp_mask = labels == comp
        n = int(comp_mask.sum())
        if n == 0:
            continue
        n_empty = int((alpha[comp_mask] < near_zero).sum())
        pct = 100.0 * n_empty / n
        if pct >= keep_pct:
            continue
        fill = comp_mask & (alpha < 255)
        out[fill] = 255
    return out


def fix_rgba(rgba: np.ndarray, flood_thresh: int = 200, edge_dilate: int = 6,
             keep_pct: int = 20) -> np.ndarray:
    fixed = rgba.copy()
    fixed[..., 3] = fill_internal_holes(rgba[..., 3], flood_thresh=flood_thresh,
                                        edge_dilate=edge_dilate, keep_pct=keep_pct)
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
           status: str, preview_fix: bool, flood_thresh: int, edge_dilate: int,
           keep_pct: int) -> np.ndarray:
    shown = fix_rgba(rgba, flood_thresh, edge_dilate, keep_pct) if preview_fix else rgba
    h, w = shown.shape[:2]
    bg = make_background(h, w, style)
    rgb_opaque = shown[..., :3]
    comp = composite(shown, bg)
    alpha_vis = np.repeat(shown[..., 3:4], 3, axis=-1)
    gap = np.full((h, 4, 3), 128, dtype=np.uint8)
    panel = np.concatenate([rgb_opaque, gap, comp, gap, alpha_vis], axis=1)
    panel_bgr = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
    return annotate(panel_bgr, [
        f"[{idx + 1}/{total}] {clip.name}",
        f"status: {status}  bg:{style % 3}  preview:{'on' if preview_fix else 'off'}  flood<{flood_thresh}  dilate={edge_dilate}  keep>={keep_pct}%",
        "n=next p=prev r=resample s=bg f=preview-fix k=keep t=throw a=apply-fix q=quit",
    ])


def render_confirm(rgba: np.ndarray, style: int, clip: Path, flood_thresh: int,
                   edge_dilate: int, keep_pct: int) -> np.ndarray:
    fixed = fix_rgba(rgba, flood_thresh, edge_dilate, keep_pct)
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
        f"CONFIRM FIX — {clip.name}   flood<{flood_thresh}  dilate={edge_dilate}  keep>={keep_pct}%",
        "y / enter = apply to whole clip   any other key = cancel",
        "r = resample a different frame first   (sliders update live)",
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

    state = {
        "idx": 0,
        "style": 0,
        "preview": False,
        "mode": "scan",  # "scan" or "confirm"
        "flood_thresh": 200,
        "edge_dilate": 6,
        "keep_pct": 20,
    }
    rgba = load_random_frame(clips[state["idx"]])

    win = "alpha-scan"
    ctrl_win = "controls  (flood< / dilate / keep%)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.namedWindow(ctrl_win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(ctrl_win, 700, 140)
    # tiny placeholder so the controls window has drawable area
    cv2.imshow(ctrl_win, np.full((1, 700, 3), 40, dtype=np.uint8))

    def _fit(img: np.ndarray) -> np.ndarray:
        if img.shape[1] > args.max_width:
            scale = args.max_width / img.shape[1]
            img = cv2.resize(img, (args.max_width, int(img.shape[0] * scale)),
                             interpolation=cv2.INTER_AREA)
        return img

    def redraw() -> None:
        clip = clips[state["idx"]]
        if state["mode"] == "confirm":
            img = render_confirm(rgba, state["style"], clip,
                                 state["flood_thresh"], state["edge_dilate"],
                                 state["keep_pct"])
        else:
            img = render(rgba, state["style"], clip, state["idx"], len(clips),
                         verdicts.get(str(clip), "unrated"),
                         state["preview"] or state["mode"] == "confirm",
                         state["flood_thresh"], state["edge_dilate"],
                         state["keep_pct"])
        cv2.imshow(win, _fit(img))

    def _on_flood(v: int) -> None:
        state["flood_thresh"] = max(1, v)
        redraw()

    def _on_dilate(v: int) -> None:
        state["edge_dilate"] = v
        redraw()

    def _on_keep(v: int) -> None:
        state["keep_pct"] = v
        redraw()

    cv2.createTrackbar("flood<", ctrl_win, state["flood_thresh"], 255, _on_flood)
    cv2.createTrackbar("dilate", ctrl_win, state["edge_dilate"], 30,  _on_dilate)
    cv2.createTrackbar("keep%",  ctrl_win, state["keep_pct"],    100, _on_keep)

    def advance() -> None:
        nonlocal rgba
        state["idx"] = (state["idx"] + 1) % len(clips)
        state["preview"] = False
        state["mode"] = "scan"
        rgba = load_random_frame(clips[state["idx"]])

    def wait_key() -> int:
        """Poll for a key so ^C and window-close events are handled promptly."""
        while True:
            k = cv2.waitKey(30)
            if k != -1:
                return k & 0xFF
            try:
                if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                    return ord("q")
            except cv2.error:
                return ord("q")

    redraw()
    try:
        while True:
            key = wait_key()
            clip = clips[state["idx"]]

            if key in (ord("q"), 27):
                break
            elif key in (ord("n"), ord(" "), 83):
                advance()
            elif key in (ord("p"), 81):
                state["idx"] = (state["idx"] - 1) % len(clips)
                state["preview"] = False
                rgba = load_random_frame(clips[state["idx"]])
            elif key == ord("r"):
                rgba = load_random_frame(clip)
            elif key == ord("s"):
                state["style"] += 1
            elif key == ord("f"):
                state["preview"] = not state["preview"]
            elif key == ord("k"):
                verdicts[str(clip)] = "keep"
                advance()
            elif key == ord("t"):
                rejected_dir.mkdir(parents=True, exist_ok=True)
                dest = rejected_dir / clip.name
                shutil.move(str(clip), str(dest))
                verdicts[str(clip)] = f"thrown -> {dest}"
                clips.pop(state["idx"])
                if not clips:
                    break
                state["idx"] %= len(clips)
                state["preview"] = False
                rgba = load_random_frame(clips[state["idx"]])
            elif key == ord("a"):
                state["mode"] = "confirm"
                confirmed = False
                while True:
                    redraw()
                    ckey = wait_key()
                    if ckey == ord("r"):
                        rgba = load_random_frame(clip)
                        continue
                    if ckey in (ord("q"), 27):
                        state["mode"] = "scan"
                        raise KeyboardInterrupt
                    confirmed = ckey in (ord("y"), 13, 10)
                    break
                state["mode"] = "scan"
                if not confirmed:
                    print("[fix] cancelled")
                else:
                    print(f"[fix] loading {clip.name} ...")
                    names, frames = load_all_frames(clip)
                    print(f"[fix] filling holes across {len(frames)} frames "
                          f"(flood<{state['flood_thresh']}, dilate={state['edge_dilate']}, "
                          f"keep%>={state['keep_pct']}) ...")
                    fixed = [fix_rgba(f, state["flood_thresh"], state["edge_dilate"],
                                      state["keep_pct"]) for f in frames]
                    out_path = write_fixed_clip(clip, names, fixed, fixed_dir)
                    print(f"[fix] wrote -> {out_path}")
                    verdicts[str(clip)] = f"fixed -> {out_path}"
                    advance()

            if not clips:
                break
            redraw()
    except KeyboardInterrupt:
        print("\ninterrupted")

    cv2.destroyAllWindows()
    for _ in range(4):
        cv2.waitKey(1)
    if verdicts:
        args.results.write_text("\n".join(f"{v}\t{p}" for p, v in verdicts.items()) + "\n")
        n_keep = sum(1 for v in verdicts.values() if v == "keep")
        n_throw = sum(1 for v in verdicts.values() if v.startswith("thrown"))
        n_fix = sum(1 for v in verdicts.values() if v.startswith("fixed"))
        print(f"saved {len(verdicts)} verdicts  keep={n_keep} thrown={n_throw} fixed={n_fix}")


if __name__ == "__main__":
    main()
