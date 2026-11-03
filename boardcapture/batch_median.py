"""Batch-median board capture (new strategy).

Pass 1 (cheap, small greyscale):
  - sample one frame every `step_s` seconds (default 2s)
  - rolling median over `batch` consecutive samples (default 15 = 30s), sliding 1 sample
    at a time -> the professor disappears because he moves and the board doesn't
  - change score = fraction of pixels that differ between consecutive rolling medians;
    a spike means the board really changed (erase, new board, slide change).
    Polarity-agnostic, so chalkboards, whiteboards and slides all work.
  - spikes that revert (prof stood still, then left) are ignored as transients
Decide offline:
  - each change point closes a "board state"; the 15-sample window that ends just
    before the change is the fullest, cleanest view of that state
Pass 2 (full resolution, only the chosen windows):
  - full-res median of that window (prof removed) + best real frame in it
    (least different from the median = least occluded, then sharpest)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path

import cv2
import numpy as np


@dataclass
class BMConfig:
    step_s: float = 2.0          # seconds between samples
    batch: int = 15              # samples per median window (15 x 2s = 30s)
    analysis_width: int = 400    # downscale for pass 1
    pixel_delta: int = 28        # grey-level change that counts as "changed"
    change_k: float = 6.0        # adaptive threshold = median + k * MAD
    change_floor: float = 0.02   # a change must touch at least 2% of pixels
    min_state: int = 3           # shortest board state worth keeping (samples)
    recent: int = 5              # final board: median of the last 5 clear frames per pixel
    occlusion_tol: int = 35      # per-pixel diff vs median that counts as occluded
    min_ink: float = 0.003       # skip blank boards
    keep_ratio: float = 0.85     # >=85% of old ink survives -> addition, not a new board


@dataclass
class Selection:
    board: int
    t_start: float
    t_end: float
    window: tuple
    reason: str
    ink: float = 0.0
    best_frame_t: float = 0.0
    best_occlusion: float = 0.0
    median_path: str = ""
    best_path: str = ""


def _frames(path, step_s, wanted=None):
    """Yield (sample_index, t, frame). If `wanted` is a set, only decode those samples."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, int(round(fps * step_s)))
    idx = 0
    while True:
        k, on_sample = divmod(idx, stride)
        if on_sample == 0 and (wanted is None or k in wanted):
            ok, frame = cap.read()
            if not ok:
                break
            yield k, idx / fps, frame
            if wanted is not None and k >= max(wanted):
                break
        elif not cap.grab():
            break
        idx += 1
    cap.release()


def _small_gray(frame, width):
    h, w = frame.shape[:2]
    g = cv2.resize(frame, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(cv2.cvtColor(g, cv2.COLOR_BGR2GRAY), (5, 5), 0)


def _small_bgr(frame, width):
    h, w = frame.shape[:2]
    return cv2.resize(frame, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)


def _prof_mask(bgr, color_dist=45):
    """Big solid blobs that are not board-coloured = person. Thin strokes (writing) survive
    the opening only if they're huge, so ink is kept; static big blobs (slide title bars)
    get masked in EVERY frame and fall back to the plain median."""
    q = (bgr >> 4).reshape(-1, 3).astype(np.int32)
    code = q[:, 0] * 256 + q[:, 1] * 16 + q[:, 2]
    mode = np.bincount(code).argmax()
    board = (np.array([mode // 256, (mode // 16) % 16, mode % 16]) * 16 + 8).astype(np.float32)
    diff = np.linalg.norm(bgr.astype(np.float32) - board, axis=2) > color_dist
    k = max(5, bgr.shape[1] // 32)
    m = cv2.morphologyEx(diff.astype(np.uint8), cv2.MORPH_OPEN,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return m.astype(bool)


def masked_median(frames, masks, recent: int | None = None):
    """Per-pixel median over frames where that pixel is NOT covered by the person.
    Falls back to the plain median where fewer than 2 frames are clear."""
    st = np.stack(frames)
    mk = np.stack(masks)
    plain = np.median(st, axis=0)
    if recent:   # use only the last `recent` clear frames per pixel -> latest writing survives
        clr = ~mk
        from_end = np.cumsum(clr[::-1], axis=0)[::-1]
        mk = ~(clr & (from_end <= recent))
    clear = (~mk).sum(axis=0)
    out = np.empty_like(plain)
    for c in range(st.shape[3]) if st.ndim == 4 else [None]:
        a = (st[..., c] if c is not None else st).astype(np.float32)
        a[mk] = np.nan
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mm = np.nanmedian(a, axis=0)
        pc = plain[..., c] if c is not None else plain
        res = np.where(clear >= 2, mm, pc)
        if c is None:
            out = res
        else:
            out[..., c] = res
    return out.astype(np.uint8)


def _ink(gray):
    bg = cv2.medianBlur(gray, 31)
    return float((cv2.absdiff(gray, bg) > 30).mean())


def _ink_mask(gray):
    bg = cv2.medianBlur(gray, 31)
    return cv2.absdiff(gray, bg) > 30


def _retention(before, after):
    """Share of the old ink still present afterwards. High = prof only added writing."""
    mb = _ink_mask(before)
    ma = cv2.dilate(_ink_mask(after).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    return float((mb & ma).sum() / max(1, mb.sum()))


def _changed(a, b, delta):
    return float((cv2.absdiff(a, b) > delta).mean())


def run(video: str, out_dir: str, cfg: BMConfig | None = None, keep_all: bool = False):
    cfg = cfg or BMConfig()
    out = Path(out_dir)
    (out / "boards").mkdir(parents=True, exist_ok=True)

    # ---------- pass 1: small greyscale samples ----------
    times, small, masks = [], [], []
    for _, t, f in _frames(video, cfg.step_s):
        times.append(t)
        small.append(_small_gray(f, cfg.analysis_width))
        masks.append(_prof_mask(_small_bgr(f, cfg.analysis_width)))
    n, w = len(small), cfg.batch
    if n < w:
        raise RuntimeError(f"video too short: {n} samples, need {w}")
    # rolling person-masked median m[k] covers samples k .. k+w-1
    med = np.stack([masked_median(small[k:k + w], masks[k:k + w])
                    for k in range(n - w + 1)])
    d = np.zeros(len(med))
    for k in range(1, len(med)):
        d[k] = _changed(med[k], med[k - 1], cfg.pixel_delta)

    mad = float(np.median(np.abs(d[1:] - np.median(d[1:]))))
    thr = max(cfg.change_floor, float(np.median(d[1:])) + cfg.change_k * mad)

    # ---------- find change points ----------
    spikes = [k for k in range(1, len(d)) if d[k] > thr]
    merged = []                                    # merge spikes within 3 samples
    for k in spikes:
        if merged and k - merged[-1][-1] <= 3:
            merged[-1].append(k)
        else:
            merged.append([k])
    changes, transients, additions = [], [], []
    for grp in merged:
        k0, k1 = grp[0], grp[-1]
        before = med[k0 - 1]
        after = med[min(len(med) - 1, k1 + w)]
        if _changed(before, after, cfg.pixel_delta) < thr / 2:
            transients.append(k0)                  # came back: prof paused, not a board change
            continue
        if _retention(before, after) >= cfg.keep_ratio:
            additions.append(k0)                   # more writing on the same board
            continue
        # median flips when the new state is the majority of the window,
        # so the real change happened about half a window later
        changes.append(int(max(grp, key=lambda k: d[k]) + w // 2))

    # ---------- choose one window per board state ----------
    selections = []
    bounds = [0] + changes + [n]
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b - a < cfg.min_state:
            continue
        lo = max(a, b - w)                         # last (up to) 15 samples of the state
        win = (lo, b - 1)
        reason = "end of video" if b == n else f"board changed at ~{times[b]:.0f}s"
        selections.append(Selection(board=0, t_start=times[a], t_end=times[b - 1],
                                    window=win, reason=reason))

    # ---------- pass 2: full-res median + best real frame ----------
    wanted = {k for s in selections for k in range(s.window[0], s.window[1] + 1)}
    full = {k: (t, f) for k, t, f in _frames(video, cfg.step_s, wanted)}
    kept = []
    for s in selections:
        ks = [k for k in range(s.window[0], s.window[1] + 1) if k in full]
        frames = [full[k][1] for k in ks]
        fm = [cv2.resize(masks[k].astype(np.uint8), (frames[0].shape[1], frames[0].shape[0]),
                         interpolation=cv2.INTER_NEAREST).astype(bool) for k in ks]
        m = masked_median(frames, fm, recent=cfg.recent)
        ms = _small_gray(m, cfg.analysis_width)
        s.ink = _ink(ms)
        if s.ink < cfg.min_ink:
            continue
        static = np.logical_and.reduce([masks[k] for k in ks])   # masked in every frame = content
        scored = []
        for k, f in zip(ks, frames):
            g = _small_gray(f, cfg.analysis_width)
            occl = float((masks[k] & ~static).mean())          # share of frame covered by the person
            sharp = float(cv2.Laplacian(g, cv2.CV_64F).var())
            scored.append((occl, -sharp, k, f))
        occl, _, bk, bf = min(scored, key=lambda x: x[:2])
        s.board = len(kept) + 1
        s.best_frame_t, s.best_occlusion = full[bk][0], occl
        s.median_path = str(out / "boards" / f"board_{s.board:02d}_median.png")
        s.best_path = str(out / "boards" / f"board_{s.board:02d}_best.png")
        cv2.imwrite(s.median_path, m)
        cv2.imwrite(s.best_path, bf)
        kept.append(s)

    if keep_all:
        (out / "all_medians").mkdir(exist_ok=True)
        for k in range(0, len(med), w):
            cv2.imwrite(str(out / "all_medians" / f"window_{k:04d}_{times[k]:.0f}s.png"), med[k])

    _contact_sheet(kept, out / "contact_sheet.jpg")
    _change_plot(times, d, thr, changes, transients, kept, out / "change_plot.png")
    report = {"video": video, "config": asdict(cfg), "samples": n, "threshold": thr,
              "changes_s": [times[c] for c in changes if c < n],
              "transients_s": [times[k] for k in transients],
              "additions_s": [times[k + w // 2] for k in additions if k + w // 2 < n],
              "selections": [asdict(s) for s in kept]}
    (out / "report.json").write_text(json.dumps(report, indent=2))
    return report


def _contact_sheet(sel, path, tile_w=480):
    rows = []
    for s in sel:
        row = []
        for p, label in ((s.median_path, "median"), (s.best_path, f"best {s.best_frame_t:.0f}s")):
            im = cv2.imread(p)
            h, w = im.shape[:2]
            im = cv2.resize(im, (tile_w, int(h * tile_w / w)))
            cv2.rectangle(im, (0, 0), (tile_w, 26), (0, 0, 0), -1)
            cv2.putText(im, f"#{s.board} {label}   state {s.t_start:.0f}-{s.t_end:.0f}s",
                        (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            row.append(im)
        rows.append(np.hstack(row))
    if rows:
        cv2.imwrite(str(path), np.vstack(rows))


def _change_plot(times, d, thr, changes, transients, sel, path, W=1000, H=280):
    img = np.full((H, W, 3), 255, np.uint8)
    n = len(d)
    top = max(float(d.max()), thr) * 1.15 or 1
    tmax = times[-1] or 1
    tx = lambda t: int(50 + t / tmax * (W - 70))
    y = lambda v: int(H - 35 - v / top * (H - 60))
    # x of rolling-median k = centre time of its window
    off = (len(times) - n) // 2
    xs = [tx(times[k + off]) for k in range(n)]
    for s in sel:
        cv2.rectangle(img, (tx(s.t_end - (s.window[1] - s.window[0]) * 2), 30),
                      (tx(s.t_end), H - 35), (210, 240, 210), -1)
    cv2.line(img, (50, y(thr)), (W - 20, y(thr)), (0, 0, 220), 1)
    cv2.putText(img, f"threshold {thr:.1%}", (W - 160, y(thr) - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 220), 1)
    for k in range(2, n):
        cv2.line(img, (xs[k - 1], y(d[k - 1])), (xs[k], y(d[k])), (50, 50, 50), 2)
    for c in changes:
        if c < len(times):
            cv2.line(img, (tx(times[c]), 30), (tx(times[c]), H - 35), (0, 0, 220), 1)
    for k in transients:
        cv2.circle(img, (xs[k], y(d[k])), 5, (0, 160, 230), -1)
    for t in range(0, int(tmax) + 1, max(30, int(tmax / 10) // 30 * 30)):
        cv2.putText(img, f"{t}s", (tx(t) - 12, H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    cv2.putText(img, "median change per step  |  red = board change  green = window kept  "
                "orange = ignored transient", (50, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)
    cv2.imwrite(str(path), img)
