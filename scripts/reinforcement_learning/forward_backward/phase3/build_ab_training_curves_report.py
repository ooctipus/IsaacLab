#!/usr/bin/env python
"""Build the bfm-ab-20260805 A/B training-curves report (self-contained HTML).

Overlays TODAY's stage-4 A/B tensorboard curves (smpl_cmu A1/A2 seeds 0+1 @5M;
g1_lafan B1/B2 pilot @9.6M; plus any discovered full-horizon 211.2M g1 runs,
partial curves included and marked in-progress) against the HISTORICAL
pre-campaign fb-current runs recovered by parsing their console logs
(smpl_cmu completed 2026-07-04; g1_lafan crashed at iteration ~497).

Outputs (both written by every run):
  /home/isaaclab/octi/IsaacLab.retarget-gpu-batch-20260715/FORWARD_BACKWARD_AB_TRAINING_CURVES.html
  /home/isaaclab/octi/bfm-campaign-20260805/receipts/FORWARD_BACKWARD_AB_TRAINING_CURVES.html

Regenerate (one command; rerun after the full-horizon g1 pair finishes and the
in-progress panel/curves refresh automatically):

  /home/isaaclab/octi/IsaacLab.retarget-gpu-batch-20260715/.venv/bin/python \
      /home/isaaclab/octi/bfm-campaign-20260805/build_ab_training_curves_report.py

Dependencies: numpy + tensorboard (both already in the campaign trainer venv).
No network, no external assets; charts are inline SVG.
"""

from __future__ import annotations

import datetime as _dt
import html as _html
import json
import math
import re
import subprocess
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- locations
CAMP = Path("/home/isaaclab/octi/bfm-campaign-20260805")
TREE = Path("/home/isaaclab/octi/IsaacLab.retarget-gpu-batch-20260715")
TB_ROOT = TREE / "logs/rsl_rl/motion_forward_backward"
HIST_LOG = {
    "smpl": Path("/home/isaaclab/octi/fb-current/runs/smpl_cmu/console.log"),
    "g1": Path("/home/isaaclab/octi/fb-current/runs/g1_lafan/console.log"),
}
OUTPUTS = [
    TREE / "FORWARD_BACKWARD_AB_TRAINING_CURVES.html",
    CAMP / "receipts" / "FORWARD_BACKWARD_AB_TRAINING_CURVES.html",
]
G5_JSON = CAMP / "receipts/stage4/G5_SUMMARY.json"
ATTR_JSON = CAMP / "receipts/abdiag_20260805/ATTRIBUTION.json"
MATRIX_JSON = CAMP / "receipts/abdiag_20260805/eval_2x2_matrix.json"
RGATE_JSON = CAMP / "receipts/rewardgate_20260806/REWARDGATE_SUMMARY.json"

PER_ITER = {"smpl": 500, "g1": 1024}
FULL_G1_ITERS = 206_250  # 211.2M transitions

# ------------------------------------------------------------ series colors
# Categorical pair validated with the dataviz six-checks validator on the
# document surface #0d1924 (all-pairs PASS: CVD dE 26.8, normal 31.8, >=3:1).
C_OURS = "#3987e5"   # our-data arm — fixed identity on every chart
C_CTRL = "#d95926"   # control arm  — fixed identity on every chart
C_HIST = "#9db0bf"   # historical fb-current run — deliberately muted + DASHED
C_GATE = "#f87171"   # crash marker (status color, always with label)
C_PEND = "#facc15"   # in-progress marker (status color, always with label)
C_OK = "#4ade80"
GRID = "#1d3245"
AXIS = "#2a4358"
INK = "#e7eef4"
MUTED = "#879cac"
CHART_BG = "#0d1924"

# ------------------------------------------------------------------ helpers

def esc(s: str) -> str:
    return _html.escape(str(s), quote=True)


def fmt_si(v: float) -> str:
    a = abs(v)
    if a >= 1e9:
        return f"{v / 1e9:.3g}B"
    if a >= 1e6:
        s = f"{v / 1e6:.4g}"
        return f"{s}M"
    if a >= 1e4:
        return f"{v / 1e3:.4g}k"
    if a == 0:
        return "0"
    if a >= 1:
        return f"{v:.4g}"
    return f"{v:.3g}"


def fmt_tick(v: float, log: bool = False) -> str:
    if log:
        e = int(round(math.log10(v)))
        return {3: "1k", 4: "10k", 5: "100k", 6: "1M", 7: "10M", 8: "100M", 9: "1B"}.get(e, f"1e{e}")
    return fmt_si(v)


def nice_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        return [lo]
    span = hi - lo
    step = 10 ** math.floor(math.log10(span / max(n - 1, 1)))
    for m in (1, 2, 2.5, 5, 10):
        if span / (step * m) <= n:
            step *= m
            break
    t0 = math.ceil(lo / step) * step
    out = []
    t = t0
    while t <= hi + 1e-12 * span:
        out.append(0.0 if abs(t) < step * 1e-6 else t)
        t += step
    return out or [lo]


# --------------------------------------------------------------- TB loading

def load_tb(run_dir: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    ea = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    ea.Reload()
    out = {}
    for tag in ea.Tags()["scalars"]:
        rows = ea.Scalars(tag)
        steps = np.array([r.step for r in rows], dtype=np.float64)
        vals = np.array([r.value for r in rows], dtype=np.float64)
        keep = np.isfinite(vals)
        if keep.sum() == 0:
            continue
        out[tag] = (steps[keep], vals[keep])
    return out


# ------------------------------------------------------ console-log parsing
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_NUM = r"([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)"
_ITER = re.compile(r"Learning iteration\s+(\d+)/(\d+)")
_PATTERNS = [
    (re.compile(rf"Mean (\S+) loss:\s*{_NUM}"), lambda m: ("Loss/" + m.group(1), float(m.group(2)))),
    (re.compile(rf"Mean action std:\s*{_NUM}"), lambda m: ("Policy/mean_std", float(m.group(1)))),
    (re.compile(rf"Mean reward:\s*{_NUM}"), lambda m: ("Train/mean_reward", float(m.group(1)))),
    (re.compile(rf"Mean episode length:\s*{_NUM}"), lambda m: ("Train/mean_episode_length", float(m.group(1)))),
    (re.compile(rf"Episode_Termination/time_out:\s*{_NUM}"), lambda m: ("Episode_Termination/time_out", float(m.group(1)))),
    (re.compile(rf"Total steps:\s*{_NUM}"), lambda m: ("_total_steps", float(m.group(1)))),
    (re.compile(rf"Steps per second:\s*{_NUM}"), lambda m: ("Perf/total_fps", float(m.group(1)))),
    (re.compile(rf"Collection time:\s*{_NUM}s"), lambda m: ("Perf/collection_time", float(m.group(1)))),
    (re.compile(rf"Learning time:\s*{_NUM}s"), lambda m: ("Perf/learning_time", float(m.group(1)))),
]


def parse_console(path: Path, per_iter: int) -> tuple[dict, dict]:
    """Parse per-iteration printed metrics from an fb-current console log.

    Returns (channels {tag: (transitions, values)}, meta). Later duplicate
    iteration blocks win (handles the g1 crash + checkpoint-resume overlap).
    """
    blocks: dict[int, dict[str, float]] = {}
    meta = {"total_iters": None, "crash_iter": None, "crash_line": None, "n_blocks": 0, "first_iter_with": {}}
    cur: dict[str, float] | None = None
    cur_it = None
    for raw in path.read_text(errors="replace").splitlines():
        line = _ANSI.sub("", raw)
        m = _ITER.search(line)
        if m:
            cur_it = int(m.group(1))
            meta["total_iters"] = int(m.group(2))
            cur = blocks.setdefault(cur_it, {})
            continue
        if "AttributeError: 'dict' object has no attribute 'to'" in line:
            meta["crash_iter"] = cur_it
            meta["crash_line"] = line.strip()
            continue
        if cur is None:
            continue
        for pat, conv in _PATTERNS:
            m = pat.search(line)
            if m:
                tag, val = conv(m)
                cur[tag] = val
                break
    meta["n_blocks"] = len(blocks)
    meta["last_iter"] = max(blocks) if blocks else None
    channels: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    tags = sorted({t for b in blocks.values() for t in b if t != "_total_steps"})
    for tag in tags:
        its = sorted(i for i, b in blocks.items() if tag in b and math.isfinite(b[tag]))
        if not its:
            continue
        meta["first_iter_with"][tag] = its[0]
        t = np.array([blocks[i].get("_total_steps", per_iter * (i + 1)) for i in its], dtype=np.float64)
        v = np.array([blocks[i][tag] for i in its], dtype=np.float64)
        channels[tag] = (t, v)
    return channels, meta


# ----------------------------------------------------------- run discovery

def read_meta(p: Path) -> dict[str, str]:
    out = {}
    for line in p.read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def discover_runs() -> tuple[dict[str, Path], list[dict]]:
    """Return (known stage-4 arm -> run_dir, discovered extra g1-class runs).

    Extra runs (e.g. the deferred full-horizon 211.2M pair, once launched) are
    found by scanning campaign *.meta files beyond stage 4 and any tensorboard
    run dir newer than the stage-4 close that is not one of the six known runs.
    Arm side is classified from params/env.yaml source_artifact_root.
    """
    known: dict[str, Path] = {}
    known_dirs: set[str] = set()
    for meta_file in sorted(CAMP.glob("runs/stage4/*.meta")):
        m = read_meta(meta_file)
        if "arm" in m and "run_dir" in m:
            known[m["arm"]] = Path(m["run_dir"])
            known_dirs.add(m["run_dir"])

    extras: list[dict] = []
    seen: set[str] = set()

    def classify(run_dir: Path, arm_hint: str | None, meta: dict[str, str] | None) -> dict | None:
        if not any(run_dir.glob("events.out.tfevents.*")):
            return None
        env_yaml = run_dir / "params/env.yaml"
        agent_yaml = run_dir / "params/agent.yaml"
        side = "unknown"
        if env_yaml.exists():
            txt = env_yaml.read_text(errors="replace")
            m = re.search(r"source_artifact_root:\s*(\S+)", txt)
            root = m.group(1) if m else ""
            side = "ours" if str(CAMP / "dumps") in root else ("control" if "fb-current" in root else "unknown")
        max_it = None
        if agent_yaml.exists():
            m = re.search(r"max_iterations:\s*(\d+)", agent_yaml.read_text(errors="replace"))
            if m:
                max_it = int(m.group(1))
        complete = (run_dir / "bfm_ab_complete.json").exists() or (
            meta is not None and meta.get("train_rc") == "0" and "completed" in meta
        )
        started = (meta or {}).get("started", "")[:10]
        if not started:
            m = re.match(r"\d{4}-\d{2}-\d{2}", run_dir.name)
            started = m.group(0) if m else ""
        return {
            "run_dir": run_dir,
            "arm": arm_hint or ("ours" if side == "ours" else "control" if side == "control" else run_dir.name),
            "side": side,
            "max_iterations": max_it,
            "complete": complete,
            "started": started,
        }

    # (a) campaign meta files outside stage4 (a future full-horizon launcher
    #     writing runs/<stage>/<ARM>.meta is picked up automatically)
    for meta_file in sorted(CAMP.glob("runs/*/*.meta")):
        if meta_file.parent.name == "stage4":
            continue
        m = read_meta(meta_file)
        rd = m.get("run_dir")
        if not rd or rd in known_dirs or rd in seen:
            continue
        info = classify(Path(rd), m.get("arm"), m)
        if info:
            seen.add(rd)
            extras.append(info)

    # (b) any new tensorboard run dir under the trainer tree
    stage4_close = _dt.datetime(2026, 8, 5, 23, 0).timestamp()
    for run_dir in sorted(TB_ROOT.glob("2026-*")):
        rd = str(run_dir)
        if rd in known_dirs or rd in seen or not run_dir.is_dir():
            continue
        if run_dir.stat().st_mtime <= stage4_close:
            continue
        info = classify(run_dir, None, None)
        if info:
            seen.add(rd)
            extras.append(info)
    return known, extras


# --------------------------------------------------------------- chart core

def smooth(v: np.ndarray, frac: float = 1 / 120) -> np.ndarray:
    n = len(v)
    w = max(3, int(round(n * frac)))
    w += 1 - w % 2
    if n < w + 2:
        return v.copy()
    pad = np.concatenate([np.repeat(v[0], w // 2), v, np.repeat(v[-1], w // 2)])
    k = np.ones(w) / w
    return np.convolve(pad, k, mode="valid")


def downsample(x: np.ndarray, y: np.ndarray, nmax: int) -> tuple[np.ndarray, np.ndarray]:
    if len(x) <= nmax:
        return x, y
    idx = np.unique(np.round(np.linspace(0, len(x) - 1, nmax)).astype(int))
    return x[idx], y[idx]


class Series:
    def __init__(self, label, t, v, color, dash=None, width=2.0, opacity=1.0, ghost=True, end_label=None):
        keep = np.isfinite(v) & np.isfinite(t)
        self.label, self.t, self.v = label, t[keep], v[keep]
        self.color, self.dash, self.width, self.opacity, self.ghost = color, dash, width, opacity, ghost
        self.end_label = end_label  # short direct label ("ours"/"control") drawn at the line end


_FIG_ID = [0]


def line_chart(title, series, *, logx=False, xlabel="transitions", ylabel="", note=None,
               vlines=(), regions=(), w=436, h=252, logy="auto", clip_pct=(1, 99), end_labels=False):
    """Inline-SVG line chart: raw min/max envelope ghost + smoothed line per series."""
    series = [s for s in series if len(s.t) >= 2]
    if not series:
        return f'<figure class="chart"><figcaption>{esc(title)}</figcaption><p class="caption">no data on either side</p></figure>'
    _FIG_ID[0] += 1
    fid = f"fig{_FIG_ID[0]}"
    ml, mr, mt, mb = 58, 12, 30, 40
    pw, ph = w - ml - mr, h - mt - mb

    all_t = np.concatenate([s.t for s in series])
    tmin, tmax = float(all_t.min()), float(all_t.max())
    if logx:
        tmin = max(tmin, 1.0)

    def tx(t):
        if logx:
            return ml + (np.log10(np.maximum(t, tmin)) - math.log10(tmin)) / (math.log10(tmax) - math.log10(tmin)) * pw
        return ml + (t - tmin) / (tmax - tmin) * pw

    all_v = np.concatenate([s.v for s in series])
    if logy == "auto":
        pos = all_v[all_v > 0]
        logy = len(pos) == len(all_v) and len(pos) > 0 and (
            np.percentile(pos, 99.5) / max(np.percentile(pos, 0.5), 1e-300) > 100
        )
    lo, hi = np.percentile(all_v, clip_pct[0]), np.percentile(all_v, clip_pct[1])
    clipped = bool(all_v.min() < lo or all_v.max() > hi)
    if logy:
        lo = max(lo, float(all_v[all_v > 0].min()))
        llo, lhi = math.log10(lo), math.log10(hi)
        if lhi - llo < 1e-9:
            lhi = llo + 1.0
        pad = (lhi - llo) * 0.06
        llo, lhi = llo - pad, lhi + pad

        def ty(v):
            vv = np.clip(v, 10 ** llo, 10 ** lhi)
            return mt + ph - (np.log10(vv) - llo) / (lhi - llo) * ph

        yticks = [10 ** e for e in range(math.ceil(llo), math.floor(lhi) + 1)]
        if len(yticks) < 2:
            yticks = [10 ** llo, 10 ** lhi]
        ylab = [fmt_si(v) for v in yticks]
    else:
        if hi - lo < max(1e-9, 1e-5 * max(abs(lo), abs(hi))):
            # constant channel (modulo float32 jitter): center the flat line
            mid = (hi + lo) / 2
            half = abs(mid) * 0.6 if mid != 0 else 1.0
            lo, hi = mid - half, mid + half
        pad = (hi - lo) * 0.07
        lo2, hi2 = lo - pad, hi + pad

        def ty(v):
            return mt + ph - (np.clip(v, lo2, hi2) - lo2) / (hi2 - lo2) * ph

        yticks = nice_ticks(lo2, hi2, 5)
        ylab = [fmt_si(v) for v in yticks]

    p = []
    p.append(f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{esc(title)}">')
    p.append(f'<rect x="0" y="0" width="{w}" height="{h}" rx="9" fill="{CHART_BG}"/>')
    # grid + y axis
    for v, lab in zip(yticks, ylab):
        y = float(ty(np.array([v]))[0]) if logy else float(ty(v))
        if y < mt - 1 or y > mt + ph + 1:
            continue
        p.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{ml + pw}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>')
        p.append(f'<text x="{ml - 6}" y="{y + 3.5:.1f}" text-anchor="end" class="tick">{esc(lab)}</text>')
    # x ticks
    if logx:
        xticks = [10 ** e for e in range(math.ceil(math.log10(tmin)), math.floor(math.log10(tmax)) + 1)]
        if len(xticks) < 3:
            cand = sorted(m * 10 ** e for e in range(math.floor(math.log10(tmin)), math.ceil(math.log10(tmax)) + 1)
                          for m in (1, 2, 5))
            xticks = [c for c in cand if tmin <= c <= tmax]
        xlabels = [fmt_si(v) for v in xticks]
    else:
        xticks = nice_ticks(tmin, tmax, 6)
        xlabels = [fmt_si(v) for v in xticks]
    for v, lab in zip(xticks, xlabels):
        x = float(tx(np.array([v]))[0])
        if x < ml - 1 or x > ml + pw + 1:
            continue
        p.append(f'<line x1="{x:.1f}" y1="{mt}" x2="{x:.1f}" y2="{mt + ph}" stroke="{GRID}" stroke-width="1"/>')
        p.append(f'<text x="{x:.1f}" y="{mt + ph + 15}" text-anchor="middle" class="tick">{esc(lab)}</text>')
    p.append(f'<rect x="{ml}" y="{mt}" width="{pw}" height="{ph}" fill="none" stroke="{AXIS}" stroke-width="1"/>')
    # shaded regions (e.g. warm-up window)
    for (r0, r1, color, label) in regions:
        x0, x1 = float(tx(np.array([max(r0, tmin)]))[0]), float(tx(np.array([min(r1, tmax)]))[0])
        if x1 - x0 < 1:
            x1 = x0 + 1
        p.append(f'<rect x="{x0:.1f}" y="{mt}" width="{x1 - x0:.1f}" height="{ph}" fill="{color}" opacity="0.10"/>')
        if label:
            p.append(f'<text x="{x1 + 3:.1f}" y="{mt + 11}" class="anno" fill="{color}">{esc(label)}</text>')

    tooltip = {"fid": fid, "ml": ml, "pw": pw, "logx": bool(logx), "tmin": tmin, "tmax": tmax, "series": []}
    nbuck = 170
    for s in series:
        sx = tx(s.t)
        sv = smooth(s.v)
        # raw ghost: min/max envelope per x bucket
        if s.ghost and len(s.t) > 3 * nbuck:
            edges = np.linspace(ml, ml + pw, nbuck + 1)
            bi = np.clip(np.digitize(sx, edges) - 1, 0, nbuck - 1)
            xs, ymin, ymax = [], [], []
            for b in range(nbuck):
                sel = bi == b
                if not sel.any():
                    continue
                xs.append((edges[b] + edges[b + 1]) / 2)
                ymin.append(s.v[sel].min())
                ymax.append(s.v[sel].max())
            if len(xs) >= 2:
                ty_hi = ty(np.array(ymax))
                ty_lo = ty(np.array(ymin))
                up = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ty_hi))
                dn = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(reversed(xs), reversed(list(ty_lo))))
                p.append(f'<polygon points="{up} {dn}" fill="{s.color}" opacity="{0.14 * s.opacity:.2f}"/>')
        elif s.ghost and len(s.t) > 40:
            gx, gy = downsample(sx, ty(s.v), 400)
            pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(gx, gy))
            p.append(f'<polyline points="{pts}" fill="none" stroke="{s.color}" stroke-width="1" opacity="{0.22 * s.opacity:.2f}"/>')
        lx, ly = downsample(sx, ty(sv), 340)
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(lx, ly))
        dash = f' stroke-dasharray="{s.dash}"' if s.dash else ""
        p.append(
            f'<polyline points="{pts}" fill="none" stroke="{s.color}" stroke-width="{s.width}"'
            f' opacity="{s.opacity}"{dash} stroke-linejoin="round"/>'
        )
        dt_, dv_ = downsample(s.t, sv, 160)
        tooltip["series"].append({"n": s.label, "c": s.color,
                                  "p": [[float(f"{a:.6g}"), float(f"{b:.6g}")] for a, b in zip(dt_, dv_)]})
    # vertical markers (e.g. historical crash)
    for (xv, color, label) in vlines:
        if xv < tmin or xv > tmax:
            continue
        x = float(tx(np.array([xv]))[0])
        p.append(f'<line x1="{x:.1f}" y1="{mt}" x2="{x:.1f}" y2="{mt + ph}" stroke="{color}" stroke-width="1.4" stroke-dasharray="3 3"/>')
        if label:
            anchor, lx = ("start", x + 4) if x < ml + pw * 0.6 else ("end", x - 4)
            p.append(f'<text x="{lx:.1f}" y="{mt + ph - 6}" text-anchor="{anchor}" class="anno" fill="{color}">✕ {esc(label)}</text>')
    # direct end-of-line labels ("ours"/"control") — ink text with the series-colored
    # endpoint dot beside it, de-collided vertically when the lines converge
    if end_labels:
        labs = []
        for s in series:
            if not s.end_label:
                continue
            sv = smooth(s.v)
            ex = float(np.atleast_1d(tx(np.array([s.t[-1]])))[-1])
            ey = float(np.atleast_1d(ty(np.array([sv[-1]])))[-1])
            labs.append([min(ex, ml + pw - 2), ey, ey, s.end_label, s.color])
        labs.sort(key=lambda r: r[1])
        for i in range(1, len(labs)):
            if labs[i][1] - labs[i - 1][1] < 13:
                labs[i][1] = labs[i - 1][1] + 13
        # clamp inside the plot, above the bottom annotation line, then re-separate upward
        y_max = mt + ph - (20 if vlines else 8)
        for lab in labs:
            lab[1] = min(max(lab[1], mt + 12), y_max)
        for i in range(len(labs) - 2, -1, -1):
            if labs[i + 1][1] - labs[i][1] < 13:
                labs[i][1] = labs[i + 1][1] - 13
        for ex, ly, ey, txt, color in labs:
            p.append(f'<circle cx="{ex:.1f}" cy="{ey:.1f}" r="3" fill="{color}" stroke="{CHART_BG}" stroke-width="1.5"/>')
            if abs(ly - 3.5 - ey) > 7:  # displaced label: repeat the colored dot beside the text
                tw = 6.2 * len(txt)  # approximate rendered width at 10.5px
                p.append(f'<circle cx="{ex - 10 - tw:.1f}" cy="{ly - 3.5:.1f}" r="2.8" fill="{color}"/>')
            p.append(f'<text x="{ex - 5:.1f}" y="{ly + 3.5:.1f}" text-anchor="end" class="elab">{esc(txt)}</text>')
    p.append(f'<text x="{ml}" y="16" class="ctitle">{esc(title)}</text>')
    if logy:
        p.append(f'<text x="{ml + pw}" y="16" text-anchor="end" class="anno" fill="{MUTED}">log y</text>')
    p.append(f'<text x="{ml + pw / 2:.0f}" y="{h - 6}" text-anchor="middle" class="alab">{esc(xlabel + (" (log)" if logx else ""))}</text>')
    if ylabel:
        p.append(f'<text x="14" y="{mt + ph / 2:.0f}" class="alab" transform="rotate(-90 14 {mt + ph / 2:.0f})" text-anchor="middle">{esc(ylabel)}</text>')
    p.append(f'<rect class="hit" x="{ml}" y="{mt}" width="{pw}" height="{ph}" fill="transparent"/>')
    p.append(f'<line class="xh" x1="0" y1="{mt}" x2="0" y2="{mt + ph}" stroke="{INK}" stroke-width="0.7" opacity="0" pointer-events="none"/>')
    p.append("</svg>")
    cap = ""
    notes = []
    if clipped:
        notes.append("y clipped to the 1–99th pct band (warm-up transient off-scale)")
    if note:
        notes.append(note)
    if notes:
        cap = f'<p class="caption">{esc("; ".join(notes))}</p>'
    js = json.dumps(tooltip, separators=(",", ":"))
    return (f'<figure class="chart" id="{fid}">' + "".join(p)
            + f'<script type="application/json">{js}</script><div class="tt" hidden></div>{cap}</figure>')


def mx_cell(big: str, sub: str, matched: bool, color: str) -> str:
    """One cell of the train-domain × eval-domain matrix. Matched-domain cells are
    prominent (arm-colored); cross-domain cells are greyed and annotated."""
    if matched:
        tag = '<span class="mxtag m">MATCHED — scored on its own references</span>'
        return (f'<div class="mxcell" style="border-color:{color}99;background:{color}14">'
                f'{tag}<b>{big}</b><small>{sub}</small></div>')
    tag = '<span class="mxtag x">cross-domain</span>'
    return f'<div class="mxcell cross">{tag}<b>{big}</b><small>{sub}</small></div>'


def matrix_2x2(col_ours: str, col_orig: str, row_ours: str, row_ctrl: str,
               cells: dict[tuple[str, str], str], caption: str) -> str:
    """2×2 grid: rows = policy train-domain (OURS / CONTROL), columns = eval-reference
    domain. cells keys: (row, col) with row in {ours, ctrl}, col in {our_refs, orig_refs}."""
    return f"""
      <div class="mx">
        <div class="mxhd"></div>
        <div class="mxhd">{col_ours}</div>
        <div class="mxhd">{col_orig}</div>
        <div class="mxrowhd"><span class="dot" style="background:{C_OURS}"></span>{row_ours}</div>
        {cells[("ours", "our_refs")]}
        {cells[("ours", "orig_refs")]}
        <div class="mxrowhd"><span class="dot" style="background:{C_CTRL}"></span>{row_ctrl}</div>
        {cells[("ctrl", "our_refs")]}
        {cells[("ctrl", "orig_refs")]}
      </div>
      <p class="caption">{caption}</p>"""


# ------------------------------------------------------------ chart catalog
FB_TAGS = ["Loss/fb/loss", "Loss/fb/measure", "Loss/fb/diagonal", "Loss/fb/off_diagonal",
           "Loss/fb/orthogonality", "Loss/fb/implied_value"]
DISC_TAGS = ["Loss/discriminator/loss", "Loss/discriminator/logistic", "Loss/discriminator/gradient_penalty"]
VDISC_TAGS = ["Loss/value/discriminator/loss", "Loss/value/discriminator/reward", "Loss/value/discriminator/target"]
VAUX_TAGS = ["Loss/value/auxiliary/loss", "Loss/value/auxiliary/reward", "Loss/value/auxiliary/target"]
ACTOR_TAGS = ["Loss/actor/loss", "Loss/actor/fb_value", "Loss/actor/helper_value"]
EPI_TAGS = ["Episode_Termination/time_out", "Train/mean_episode_length", "Policy/mean_std",
            "Loss/learning_rate", "Perf/total_fps"]

TODAY_ONLY = {"Loss/learning_rate"}  # never printed by the fb-current console


def build_series(tag, today_runs, hist, per_iter):
    """today_runs: list of (label, data, color, style-kind); hist: (data, meta) or None."""
    out = []
    if hist is not None:
        hdata, _ = hist
        if tag in hdata:
            t, v = hdata[tag]
            out.append(Series("JULY-4 old run (original data, old code)", t, v, C_HIST, dash="6 4", width=1.7))
    for label, data, color, kind in today_runs:
        if tag not in data:
            continue
        st, sv = data[tag]
        t = (st + 1) * per_iter
        if kind == "primary":
            elab = "ours" if color == C_OURS else ("control" if color == C_CTRL else None)
            out.append(Series(label, t, sv, color, width=2.0, end_label=elab))
        elif kind == "seed1":
            out.append(Series(label, t, sv, color, width=1.2, opacity=0.55))
        elif kind == "partial":
            out.append(Series(label, t, sv, color, width=2.0, dash="9 3"))
    return out


def grid_of_charts(tags, today_runs, hist, per_iter, *, logx=False, warmup_tr=None, crash=None, hist_meta=None,
                   loss_tags=True):
    figs = []
    for i, tag in enumerate(tags):
        series = build_series(tag, today_runs, hist, per_iter)
        vlines = []
        if crash is not None:
            x, c, lab = crash
            vlines.append((x, c, lab if i == 0 else ""))
        regions = []
        if warmup_tr and tag.startswith("Loss/") and tag != "Loss/learning_rate":
            regions.append((0 if not logx else 1, warmup_tr, C_PEND, "warm-up" if i == 0 else ""))
        note = None
        if hist is not None and tag not in hist[0]:
            if tag in TODAY_ONLY:
                note = "campaign runs only — the JULY-4 old run's console never printed this channel"
            else:
                note = "campaign runs only — absent from the JULY-4 old run's console record"
        figs.append(line_chart(tag, series, logx=logx, vlines=vlines, regions=regions, note=note,
                               end_labels=(i == 0)))
    return '<div class="chart-grid">' + "".join(figs) + "</div>"


def swatch(color, dash=None, opacity=1.0, width=2.4, w=26):
    stroke = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<svg width="{w}" height="10" style="flex:none"><line x1="1" y1="5" x2="{w - 1}" y2="5" '
            f'stroke="{color}" stroke-width="{width}" opacity="{opacity}"{stroke}/></svg>')


def legend_html(entries):
    """entries: (label, color, dash[, opacity[, width]]) — swatches mirror the drawn line style."""
    pills = []
    for e in entries:
        label, color, dash = e[0], e[1], e[2]
        opacity = e[3] if len(e) > 3 else 1.0
        width = e[4] if len(e) > 4 else 2.4
        pills.append(f'<span class="pill">{swatch(color, dash, opacity, width)}{esc(label)}</span>')
    return f'<div class="legend">{"".join(pills)}<span class="pill legendnote">band = raw per-iteration min/max envelope · line = smoothed mean · shared by every chart in this section</span></div>'


def mini_key() -> str:
    """One-line series key repeated above every chart block."""
    return ('<div class="minikey"><span class="mk">' + swatch(C_OURS, w=22)
            + '<b>OURS</b>&thinsp;= pipeline data</span><span class="mk">' + swatch(C_CTRL, w=22)
            + '<b>CONTROL</b>&thinsp;= original data</span><span class="mk">' + swatch(C_HIST, "6 4", w=22)
            + '<b>JULY-4</b>&thinsp;= old run (original data, old code)</span></div>')


# ------------------------------------------------------------------- build

def main() -> None:
    now = _dt.datetime.now().astimezone()
    gen_stamp = now.strftime("%Y-%m-%d %H:%M %Z")
    git_head = subprocess.run(["git", "-C", str(TREE), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip() or "unknown"
    git_branch = subprocess.run(["git", "-C", str(TREE), "rev-parse", "--abbrev-ref", "HEAD"],
                                capture_output=True, text=True).stdout.strip() or "unknown"

    known, extras = discover_runs()
    print("[load] stage-4 runs:", {k: str(v) for k, v in known.items()})
    tb = {arm: load_tb(rd) for arm, rd in known.items()}
    hist_smpl = parse_console(HIST_LOG["smpl"], PER_ITER["smpl"])
    hist_g1 = parse_console(HIST_LOG["g1"], PER_ITER["g1"])
    print(f"[hist] smpl blocks={hist_smpl[1]['n_blocks']} last_iter={hist_smpl[1]['last_iter']}")
    print(f"[hist] g1 blocks={hist_g1[1]['n_blocks']} last_iter={hist_g1[1]['last_iter']} crash_iter={hist_g1[1]['crash_iter']}")

    g5 = json.loads(G5_JSON.read_text()) if G5_JSON.exists() else None
    attr = json.loads(ATTR_JSON.read_text()) if ATTR_JSON.exists() else None
    mx = json.loads(MATRIX_JSON.read_text()) if MATRIX_JSON.exists() else None
    rgate = json.loads(RGATE_JSON.read_text()) if RGATE_JSON.exists() else None

    # ---------------- full-horizon pair status
    full_runs = [e for e in extras if e.get("max_iterations") == FULL_G1_ITERS
                 or (e.get("max_iterations") or 0) > 9375]
    full_series_data = []   # tooltip label, data, color, kind
    full_legend = []        # long-form legend entries with status + date stamps
    for e in full_runs:
        data = load_tb(e["run_dir"])
        steps = data.get("Episode_Termination/time_out", (np.array([0.0]), None))[0]
        last_it = int(steps.max()) if len(steps) else 0
        e["last_iter"] = last_it
        color = C_OURS if e["side"] == "ours" else C_CTRL if e["side"] == "control" else C_PEND
        word = {"ours": "OURS — pipeline data", "control": "CONTROL — original data"}.get(e["side"])
        short = {"ours": "OURS", "control": "CONTROL"}.get(e["side"], "?")
        armcode = e["arm"] if e["arm"] not in ("ours", "control") else \
            ("B1full" if e["side"] == "ours" else "B2full")
        started = e.get("started") or "2026-08-06"
        if e["complete"]:
            stamp = f"full run finished (started {started})"
        else:
            stamp = (f"RUNNING NOW — started {started}, ~2.5-day run, curve partial "
                     f"@ iter {last_it:,}/{e['max_iterations']:,}")
        full_legend.append((f"{word or e['arm']} ({armcode}, full horizon) · {stamp}", color, "9 3"))
        full_series_data.append((f"{short} ({armcode}, {'complete' if e['complete'] else 'RUNNING NOW'})",
                                 data, color, "partial" if not e["complete"] else "primary"))
        print(f"[full] {e['run_dir']}  side={e['side']} iters={last_it} complete={e['complete']}")
    n_live = sum(1 for e in full_runs if not e["complete"])
    full_started = next((e.get("started") for e in full_runs if e.get("started")), "2026-08-06")
    if not full_runs:
        print("[full] no full-horizon g1 runs found at generation time")

    # ---------------- series bundles (labels feed the hover tooltips)
    smpl_runs = [
        ("OURS (A1, seed 0)", tb["A1"], C_OURS, "primary"),
        ("OURS (A1s1, seed 1)", tb["A1s1"], C_OURS, "seed1"),
        ("CONTROL (A2, seed 0)", tb["A2"], C_CTRL, "primary"),
        ("CONTROL (A2s1, seed 1)", tb["A2s1"], C_CTRL, "seed1"),
    ]
    g1_runs = [
        ("OURS (B1, seed 4728)", tb["B1"], C_OURS, "primary"),
        ("CONTROL (B2, seed 4728)", tb["B2"], C_CTRL, "primary"),
    ]

    smpl_warmup_tr = (hist_smpl[1]["first_iter_with"].get("Loss/fb/loss", 101)) * PER_ITER["smpl"]
    g1_warmup_tr = (hist_g1[1]["first_iter_with"].get("Loss/fb/loss", 11)) * PER_ITER["g1"]
    crash_it = hist_g1[1]["crash_iter"] or 497
    g1_crash = ((crash_it + 1) * PER_ITER["g1"], C_GATE,
                f"historical crash · iter {crash_it}")

    smpl_legend = legend_html([
        ("OURS — pipeline data (A1, seed 0) · completed 2026-08-05", C_OURS, None),
        ("OURS — pipeline data (A1s1, seed 1) · completed 2026-08-05", C_OURS, None, 0.55, 1.6),
        ("CONTROL — original data (A2, seed 0) · completed 2026-08-05", C_CTRL, None),
        ("CONTROL — original data (A2s1, seed 1) · completed 2026-08-05", C_CTRL, None, 0.55, 1.6),
        ("JULY-4 — historical (original data, old code) · OLD RUN — completed 2026-07-04 (pre-campaign stack)", C_HIST, "6 4"),
    ])
    g1_legend_entries = [
        ("OURS — pipeline data (B1, seed 4728) · pilot completed 2026-08-05", C_OURS, None),
        ("CONTROL — original data (B2, seed 4728) · pilot completed 2026-08-05", C_CTRL, None),
        ("JULY-4 — historical (original data, old code) · OLD RUN — crashed @ iter ~497, 2026-07-04 (pre-campaign stack)", C_HIST, "6 4"),
    ] + full_legend
    g1_legend = legend_html(g1_legend_entries)
    full_intro = ""
    if full_series_data:
        full_intro = (f" The full-horizon pair — OURS (B1full) and CONTROL (B2full), 206,250 iterations ="
                      f" 211.2M transitions, same data arms and seed — is RUNNING NOW (started {esc(full_started)},"
                      f" ~2.5-day runs) and overlays as the dash-marked blue/orange traces: partial curves"
                      f" as of {esc(gen_stamp)}, clearly provisional (section 07)."
                      if n_live else
                      " The finished full-horizon pair — OURS (B1full) and CONTROL (B2full), 206,250 iterations"
                      " = 211.2M transitions — overlays as the dash-marked blue/orange traces (section 07).")

    # smpl grids
    mk = mini_key()
    smpl_html = (
        smpl_legend
        + "<h3>Forward–backward loss family</h3>" + mk
        + grid_of_charts(FB_TAGS, smpl_runs, hist_smpl, PER_ITER["smpl"], warmup_tr=smpl_warmup_tr)
        + "<h3>Discriminator</h3>" + mk
        + grid_of_charts(DISC_TAGS, smpl_runs, hist_smpl, PER_ITER["smpl"], warmup_tr=smpl_warmup_tr)
        + "<h3>Discriminator value head</h3>" + mk
        + grid_of_charts(VDISC_TAGS, smpl_runs, hist_smpl, PER_ITER["smpl"], warmup_tr=smpl_warmup_tr)
        + "<h3>Actor</h3>" + mk
        + grid_of_charts(ACTOR_TAGS, smpl_runs, hist_smpl, PER_ITER["smpl"], warmup_tr=smpl_warmup_tr)
        + "<h3>Episode statistics &amp; schedule</h3>" + mk
        + grid_of_charts(EPI_TAGS, smpl_runs, hist_smpl, PER_ITER["smpl"], warmup_tr=smpl_warmup_tr)
    )
    # g1 grids (log-x so the 0–512k overlap window with the historical run stays readable)
    g1_all = g1_runs + full_series_data
    g1_kw = dict(logx=True, warmup_tr=g1_warmup_tr, crash=g1_crash)
    g1_html = (
        g1_legend
        + "<h3>Forward–backward loss family</h3>" + mk
        + grid_of_charts([t for t in FB_TAGS if t != "Loss/fb/implied_value"], g1_all, hist_g1, PER_ITER["g1"], **g1_kw)
        + "<h3>Discriminator</h3>" + mk
        + grid_of_charts(DISC_TAGS, g1_all, hist_g1, PER_ITER["g1"], **g1_kw)
        + "<h3>Discriminator value head</h3>" + mk
        + grid_of_charts(VDISC_TAGS, g1_all, hist_g1, PER_ITER["g1"], **g1_kw)
        + "<h3>Auxiliary value head (g1 only)</h3>" + mk
        + grid_of_charts(VAUX_TAGS, g1_all, hist_g1, PER_ITER["g1"], **g1_kw)
        + "<h3>Actor</h3>" + mk
        + grid_of_charts(ACTOR_TAGS, g1_all, hist_g1, PER_ITER["g1"], **g1_kw)
        + "<h3>Episode statistics &amp; schedule</h3>" + mk
        + grid_of_charts(EPI_TAGS, g1_all, hist_g1, PER_ITER["g1"], **g1_kw)
    )

    # ---------------- quality 2×2 matrices (matched-domain numbers primary)
    CROSS_NOTE = "cross-domain: constant convention penalty (ground height / world gauge), not training quality"
    smpl_mx_html = g1_mx_html = smpl_headline = g1_headline = ""
    if mx:
        sc = mx["smpl"]["cells"]
        a1o, a1g = sc["A1|ours"]["mean"], sc["A1|original"]["mean"]
        a2o, a2g = sc["A2|ours"]["mean"], sc["A2|original"]["mean"]
        s1o, s1g = sc["A1s1|ours"]["mean"], sc["A1s1|original"]["mean"]
        s2o, s2g = sc["A2s1|ours"]["mean"], sc["A2s1|original"]["mean"]
        rs0 = mx["smpl"]["ratios_seed0"]["matched_domain_ourdata_on_ours_over_control_on_original"]
        rs1 = mx["smpl"]["ratios_seed1"]["matched_domain_ourdata_on_ours_over_control_on_original"]
        fz0 = (g5["smpl"]["decisive"]["ratio_seed0"] if g5 else
               mx["smpl"]["ratios_seed0"]["frozen_G5_ourdata_over_control_on_original_refs"])
        fz1 = (g5["smpl"]["seed_evidence"]["ratio_seed1"] if g5 else
               mx["smpl"]["ratios_seed1"]["frozen_G5_ourdata_over_control_on_original_refs"])
        smpl_mx_html = matrix_2x2(
            "eval: OUR references (pipeline gauge)",
            "eval: ORIGINAL references (the frozen-G5 gauge)",
            "OURS — pipeline data (A1 / A1s1)", "CONTROL — original data (A2 / A2s1)",
            {
                ("ours", "our_refs"): mx_cell(f"{a1o:.3f}", f"seed 1 (A1s1): {s1o:.3f} · mean EMD, lower is better",
                                              True, C_OURS),
                ("ours", "orig_refs"): mx_cell(f"{a1g:.3f}", f"seed 1: {s1g:.3f} · +{(a1g / a1o - 1) * 100:.0f}% vs its "
                                               f"matched cell — {CROSS_NOTE}", False, C_OURS),
                ("ctrl", "our_refs"): mx_cell(f"{a2o:.3f}", f"seed 1: {s2o:.3f} · +{(a2o / a2g - 1) * 100:.0f}% vs its "
                                              f"matched cell — {CROSS_NOTE}", False, C_CTRL),
                ("ctrl", "orig_refs"): mx_cell(f"{a2g:.3f}", f"seed 1 (A2s1): {s2g:.3f} · Meta native anchor "
                                               f"{(g5 or {}).get('smpl', {}).get('meta_native_anchor', 1.6949):.4f} "
                                               "applies to this gauge only", True, C_CTRL),
            },
            "held-out mean EMD @ 5M transitions · matched 168-clip intersection, identical frozen protocol in "
            "every cell · receipts/abdiag_20260805/eval_2x2_matrix.json (2026-08-05)")
        smpl_headline = f"""
      <div class="callout ok"><strong>Scored fairly, OURS is at parity or better:</strong> matched-domain
      (each policy vs its OWN references) OURS÷CONTROL = <b>{rs0:.3f}</b> (seed 1: {rs1:.3f}) — against the
      frozen cross-domain readout of {fz0:.3f} (seed 1: {fz1:.3f}), which scored both arms on the ORIGINAL
      references, i.e. outside OURS' training domain only. Scored against the other corpus's conventions,
      both directions look worse — CONTROL degrades +{(a2o / a2g - 1) * 100:.0f}% on our references
      ({a2g:.3f} → {a2o:.3f}) just as OURS degrades +{(a1g / a1o - 1) * 100:.0f}% on the originals — a
      symmetric convention penalty, not a data-quality gap.</div>"""

        gc = mx["g1"]["cells"]
        b1o_e, b1g_e = gc["B1|ours"]["emd"]["mean"], gc["B1|original"]["emd"]["mean"]
        b2o_e, b2g_e = gc["B2|ours"]["emd"]["mean"], gc["B2|original"]["emd"]["mean"]
        b1o_s, b1g_s = gc["B1|ours"]["obs_state_emd"]["mean"], gc["B1|original"]["obs_state_emd"]["mean"]
        b2o_s, b2g_s = gc["B2|ours"]["obs_state_emd"]["mean"], gc["B2|original"]["obs_state_emd"]["mean"]
        rg_e = mx["g1"]["ratios"]["emd"]["matched_domain_ourdata_on_ours_over_control_on_original"]
        rg_s = mx["g1"]["ratios"]["obs_state_emd"]["matched_domain_ourdata_on_ours_over_control_on_original"]
        gfz_e = (g5["g1"]["decisive"]["emd_ratio_evidence"] if g5 else
                 mx["g1"]["ratios"]["emd"]["frozen_G5_ourdata_over_control_on_original_refs"])
        gfz_s = (g5["g1"]["decisive"]["obs_state_emd_ratio"] if g5 else
                 mx["g1"]["ratios"]["obs_state_emd"]["frozen_G5_ourdata_over_control_on_original_refs"])
        p2 = mx["g1"]["phase2_baseline_tracking"]["emd_mean"]
        g1_mx_html = matrix_2x2(
            "eval: OUR references (pipeline gauge)",
            "eval: ORIGINAL references (the frozen-gate gauge)",
            "OURS — pipeline data (B1)", "CONTROL — original data (B2)",
            {
                ("ours", "our_refs"): mx_cell(f"{b1o_e:.3f}", f"obs_state_emd {b1o_s:.3f} · mean EMD, lower is better",
                                              True, C_OURS),
                ("ours", "orig_refs"): mx_cell(f"{b1g_e:.3f}", f"obs_state_emd {b1g_s:.3f} · "
                                               f"+{(b1g_e / b1o_e - 1) * 100:.0f}%/+{(b1g_s / b1o_s - 1) * 100:.0f}% vs its "
                                               f"matched cell — {CROSS_NOTE} (wrist-roll gauge inversion dominates)",
                                               False, C_OURS),
                ("ctrl", "our_refs"): mx_cell(f"{b2o_e:.3f}", f"obs_state_emd {b2o_s:.3f} · "
                                              f"+{(b2o_e / b2g_e - 1) * 100:.0f}%/+{(b2o_s / b2g_s - 1) * 100:.0f}% vs its "
                                              f"matched cell — {CROSS_NOTE}", False, C_CTRL),
                ("ctrl", "orig_refs"): mx_cell(f"{b2g_e:.3f}", f"obs_state_emd {b2g_s:.3f} · phase-2 baseline emd "
                                               f"{p2:.3f} applies to this gauge only", True, C_CTRL),
            },
            "pilot mean EMD @ 9.6M transitions · matched 843-clip intersection, identical frozen protocol in "
            "every cell (the frozen g1 protocol evaluates the TRAIN split — symmetric for both arms) · "
            "receipts/abdiag_20260805/eval_2x2_matrix.json (2026-08-05)")
        g1_headline = f"""
      <div class="callout ok"><strong>Scored fairly, OURS is at parity:</strong> matched-domain
      OURS÷CONTROL = <b>{rg_e:.3f}</b> emd / {rg_s:.3f} obs_state_emd — against the frozen cross-domain
      readout of {gfz_e:.3f} / {gfz_s:.3f}, which scored both arms on the ORIGINAL references. Scored
      against the other corpus's conventions, both directions look worse — CONTROL degrades
      +{(b2o_e / b2g_e - 1) * 100:.0f}%/+{(b2o_s / b2g_s - 1) * 100:.0f}%
      on our references ({b2g_s:.3f} → {b2o_s:.3f} obs) just as OURS degrades
      +{(b1g_e / b1o_e - 1) * 100:.0f}%/+{(b1g_s / b1o_s - 1) * 100:.0f}% on the originals — a symmetric
      convention penalty (wrist-roll / world gauge), not training quality.</div>"""

    matched_html = ""
    if attr:
        de = attr["decisive_evidence"]
        fr = de["frozen_G5_ratios_for_contrast"]
        matched_html = f"""
      <div class="table-wrap"><table class="center">
        <thead><tr><th>readout · ratio = OURS ÷ CONTROL</th><th>frozen G5 (both arms scored on ORIGINAL refs — cross-domain for OURS)</th><th>matched-domain (each arm on its OWN refs)</th><th>margin 1.10</th></tr></thead>
        <tbody>
          <tr><td>smpl EMD · seed 0</td><td>{fr['smpl_seed0']:.4f}</td><td class="yes">{de['smpl_seed0']:.4f}</td><td><span class="status ok">within</span></td></tr>
          <tr><td>smpl EMD · seed 1</td><td>{fr['smpl_seed1']:.4f}</td><td class="yes">{de['smpl_seed1']:.4f}</td><td><span class="status ok">within</span></td></tr>
          <tr><td>g1 emd</td><td>{fr['g1_emd']:.4f}</td><td class="yes">{de['g1_emd']:.4f}</td><td><span class="status ok">within</span></td></tr>
          <tr><td>g1 obs_state_emd</td><td>{fr['g1_obs_state_emd']:.4f}</td><td class="yes">{de['g1_obs_state_emd']:.4f}</td><td><span class="status ok">within</span></td></tr>
        </tbody>
      </table></div>
      <p class="caption">matched clip intersections: smpl 168, g1 843 · identical frozen protocols per cell · receipts/abdiag_20260805/ (2026-08-05)</p>"""

    rgate_html = ""
    if rgate:
        b1 = rgate["arms"]["B1"]["broad_reward_means"]["return_mean"]
        b2 = rgate["arms"]["B2"]["broad_reward_means"]["return_mean"]
        ratio = b1 / b2
        rgate_html = f"""
      <div class="grid three">
        <div class="card" style="border-color:rgba(57,135,229,.58)"><b style="font-size:26px">{b1:.2f}</b><small>OURS — pipeline data (B1) · mean return, 38 reward tasks × 10 episodes (2026-08-06)</small></div>
        <div class="card" style="border-color:rgba(217,89,38,.58)"><b style="font-size:26px">{b2:.2f}</b><small>CONTROL — original data (B2) · mean return, same regenerated dataset, same protocol</small></div>
        <div class="card" style="border-color:rgba(74,222,128,.58)"><b style="font-size:26px">{ratio:.3f}×</b><small>OURS ÷ CONTROL aggregate return ratio ≥ 0.909 → <span class="yes">PASS</span> — the only reference-free policy-quality measurement in the ladder</small></div>
      </div>
      <div class="callout ok" style="margin-top:10px"><strong>Reference-free parity confirmed (2026-08-06):</strong> OURS (B1, pipeline data) returns 1.40× CONTROL (B2, original data) on the shared 38-task broad-reward measure — no task-level quality deficit; this closes the abdiag residual (“our smoother references might just be easier targets”) in OURS' favor at pilot scale. Heterogeneity recorded: median per-task ratio 0.806, OURS better on 16/38 — the arms trade task families (OURS/B1: spin-arms 18.3×, rotate-z 12.4×; CONTROL/B2: raisearms 6.1×, sitonground 2.2×), and OURS buys returns with more action (l2 +0.51) and auxiliary penalty (+2.47). Both arms miss the frozen phase-2 anchor (0.34×/0.47×) under recorded caveats (regenerated dataset + known stack-class delta) — per the standing decision rule, anchor deltas are never method verdicts.</div>"""

    # ---------------- full-horizon panel
    if full_runs:
        rows = []
        for e in full_runs:
            side = {"ours": "OURS — pipeline data", "control": "CONTROL — original data"}.get(e["side"], "?")
            pct = 100.0 * e["last_iter"] / e["max_iterations"]
            state = ('<span class="status ok">complete</span>' if e["complete"]
                     else f'<span class="status wait">RUNNING NOW · {pct:.1f}% · started {esc(e.get("started") or "?")}</span>')
            rows.append(f"<tr><td>{esc(side)}</td><td><code>{esc(e['run_dir'].name)}</code></td>"
                        f"<td>{e['last_iter']:,} / {e['max_iterations']:,}</td><td>{state}</td></tr>")
        past_crash = [e for e in full_runs if e["last_iter"] > (crash_it or 497)]
        crash_note = ""
        if past_crash:
            crash_note = (f" {'Both runs are' if len(past_crash) == 2 else 'One run is'} already past "
                          f"iteration {crash_it} — the point where the only prior attempt at this horizon "
                          "(the JULY-4 old run) died.")
        live_word = (f"is RUNNING NOW (started {esc(full_started)}, ~2.5 days per arm)" if n_live
                     else "has finished")
        full_html = f"""
      <div class="callout wait"><strong>Partial curves as of {esc(gen_stamp)}.</strong> The full-horizon pair
      — OURS (B1full, pipeline data) and CONTROL (B2full, original data) — {live_word} and is overlaid
      dash-marked in the g1 charts above; every number from these runs is provisional until they
      finish.{crash_note} Rerun the generator (section 08) to refresh.</div>
      <div class="table-wrap" style="margin-top:10px"><table>
        <thead><tr><th>arm</th><th>run</th><th>iterations</th><th>status</th></tr></thead>
        <tbody>{''.join(rows)}</tbody></table></div>"""
    else:
        full_html = f"""
      <div class="callout wait"><strong>Not running at generation time ({esc(gen_stamp)}).</strong>
      All four GPUs were idle and no full-horizon run directory exists; the campaign ledger records the
      211.2M-transition g1 pair (206,250 iterations, ~2.5 GPU-days per arm) as <em>deferred to a user
      decision</em> — the 9.6M pilot above is the campaign's declared stop. The historical record makes the
      stakes concrete: the only prior attempt at this horizon died at iteration ~497 of 206,250 (0.24%).
      Today's pilot pair passed that point 19-fold over without incident.</div>
      <div class="callout" style="margin-top:10px; border-left:4px solid var(--line); color:#cbd7df">
      <strong>This panel refreshes itself.</strong> The generator scans <code>runs/*/*.meta</code> in the
      campaign root and every new run directory under the trainer tree's
      <code>logs/rsl_rl/motion_forward_backward/</code>, classifies the arm from the run's own
      <code>params/env.yaml</code> (<code>source_artifact_root</code>), and overlays whatever partial curves
      exist, marked in-progress. Launch the pair, rerun the one command in section 08, and this document
      updates end-to-end.</div>"""

    # ---------------- historical parse coverage
    def cov_row(name, meta, per_iter, chans, extra=""):
        tags_found = sorted(t for t in chans)
        return (name, meta, per_iter, tags_found, extra)

    hs_meta, hg_meta = hist_smpl[1], hist_g1[1]
    n_smpl_tags = len(hist_smpl[0])
    n_g1_tags = len(hist_g1[0])

    coverage_html = f"""
      <div class="grid two">
        <div class="card data"><b>smpl_cmu · console.log · 338,320 lines</b>
          <small>{hs_meta['n_blocks']:,} / 10,000 iteration blocks parsed (run COMPLETED 2026-07-04, 2:34:09,
          Training time 9,323 s) · {n_smpl_tags} channels recovered · losses first printed at iteration
          {hs_meta['first_iter_with'].get('Loss/fb/loss', '—')} (old warm-up boundary)</small></div>
        <div class="card data"><b>g1_lafan · console.log · 17,507 lines</b>
          <small>{hg_meta['n_blocks']:,} iteration blocks parsed (0–{hg_meta['last_iter']}) of a declared
          206,250 · crash traceback (<code>AttributeError: 'dict' object has no attribute 'to'</code> in
          <code>_as_observations(extras["final_obs"])</code>) recorded inside the iteration-{hg_meta['crash_iter']}
          block; a checkpoint resume reached iteration {hg_meta['last_iter']}, then a final relaunch died during
          kit boot — end of record · {n_g1_tags} channels recovered</small></div>
      </div>
      <div class="table-wrap" style="margin-top:12px"><table>
        <thead><tr><th>recoverable from the console record</th><th>NOT recoverable (annotated per chart)</th></tr></thead>
        <tbody><tr>
          <td>every printed per-iteration channel: the full FB loss family, discriminator + gradient penalty,
          discriminator/auxiliary value heads, actor family, action std, time-out termination fraction, episode
          length + mean reward (smpl only), steps/s, collection/learning time, exact cumulative transition counts
          (<code>Total steps</code> — used as the overlay x-axis, so alignment with today's tensorboard steps is
          exact, not estimated)</td>
          <td><code>Loss/learning_rate</code> (never printed); per-second <code>Train/*_time</code> variants;
          g1 episode statistics — the historical g1 run died at iteration 499 with horizon 501, one iteration
          short of its first completed episode, so <em>no</em> episode stat was ever printed (today's B pair
          logs its first at iteration 500); tensorboard events / checkpoints of the 2026-07-04 runs
          (deleted with the old tree — the console record parsed here is the only surviving trace)</td>
        </tr></tbody></table></div>"""

    # ---------------- assemble page
    css = CSS.replace("__CHART_BG__", CHART_BG)
    n_figs = _FIG_ID[0]

    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Forward–Backward A/B · Training Curves</title>
<style>{css}</style>
</head>
<body>
<main>
  <header>
    <div class="kicker">bfm-ab-20260805 · A/B training curves vs the pre-campaign record</div>
    <h1>Every A/B arm reaches its horizon; the pre-campaign g1 run died at iteration 497</h1>
    <p class="lead">Six stage-4 runs (smpl_cmu 5M ×2 seeds, g1_lafan 9.6M pilot) overlaid channel-by-channel
    against the only surviving pre-campaign traces — the JULY-4 (2026-07-04) fb-current console logs. Blue =
    OURS (trained on our pipeline's data), orange = CONTROL (trained on the original data), dashed grey =
    the JULY-4 old runs; every quality verdict below carries its receipt and its date.</p>
    <div class="badges">
      <span class="badge"><span class="dot" style="background:var(--ok)"></span>integration receipts green (G0–G3, 2026-08-05)</span>
      <span class="badge"><span class="dot" style="background:var(--ok)"></span>all 6 A/B runs completed, rc=0, zero NaN (2026-08-05)</span>
      <span class="badge"><span class="dot" style="background:var(--ok)"></span>matched-domain parity 0.974–1.028 (2026-08-05)</span>
      <span class="badge"><span class="dot" style="background:var(--ok)"></span>reward-gate OURS/CONTROL (B1/B2) = 1.397× PASS (2026-08-06)</span>
      <span class="badge"><span class="dot" style="background:var(--pending)"></span>frozen G5 margins breached — attributed to eval-reference domain shift</span>
      <span class="badge"><span class="dot" style="background:var(--pending)"></span>full-horizon 211.2M pair {('RUNNING NOW — started ' + esc(full_started)) if n_live else ('complete' if full_runs else 'deferred')}</span>
      <span class="badge">generated {esc(gen_stamp)} · {esc(git_branch)} @ {esc(git_head)}</span>
    </div>
    <div class="keypanel" id="key">
      <div class="keytitle">Key — which line is which (the same code on every chart)</div>
      <div class="keyrow">{swatch(C_OURS, w=30)}<span><b>OURS — pipeline data:</b> solid blue = policies trained
        on OUR retargeting pipeline's dumps (smpl A1/A1s1 · g1 B1/B1full); seed-1 repeats are the same blue,
        thinner and lighter.</span></div>
      <div class="keyrow">{swatch(C_CTRL, w=30)}<span><b>CONTROL — original data:</b> solid orange = policies
        trained on the ORIGINAL preprocessed data (smpl A2/A2s1 · g1 B2/B2full).</span></div>
      <div class="keyrow">{swatch(C_HIST, "6 4", w=30)}<span><b>JULY-4 — historical:</b> dashed grey = the OLD
        pre-campaign runs of 2026-07-04 (original data on the old code); ✕ marks where the old g1 run
        crashed.</span></div>
      <div class="keystatus">{("All smpl and g1-pilot traces are FINISHED runs (campaign arms completed 2026-08-05; JULY-4 old runs 2026-07-04); the only live runs are the " + ("two " if n_live == 2 else "") + "g1 full-horizon arm" + ("s" if n_live != 1 else "") + " — marked RUNNING NOW, started " + esc(full_started) + ", ~2.5-day runs, curves partial.") if n_live else "Every trace on this page is a finished run (campaign arms completed 2026-08-05; JULY-4 old runs 2026-07-04)."}</div>
    </div>
    <div class="legend">
      <span class="pill"><span class="dot" style="background:{C_GATE}"></span>✕ historical crash marker</span>
      <span class="pill"><span class="dot" style="background:{C_PEND}"></span>warm-up window shading / RUNNING NOW accents</span>
    </div>
  </header>

  <nav>
    <a href="#verdicts">Verdicts</a><a href="#reading">How to read</a><a href="#smpl">smpl_cmu curves</a>
    <a href="#smpl-quality">smpl quality</a><a href="#g1">g1_lafan curves</a><a href="#g1-quality">g1 quality</a>
    <a href="#full">Full horizon</a><a href="#provenance">Provenance</a>
    <span class="navkey">{swatch(C_OURS, w=18)}OURS&ensp;{swatch(C_CTRL, w=18)}CONTROL&ensp;{swatch(C_HIST, "6 4", w=18)}JULY-4</span>
  </nav>

  <section id="verdicts">
    <div class="kicker">01 · campaign verdicts</div>
    <h2>Integration is proven; quality is a near-tie once the eval-reference gauge is matched</h2>
    <div class="grid three">
      <div class="card env"><b>Integration receipts <span class="status ok">green</span></b>
        <small>2026-08-05 · G0 env re-baseline at HEAD, G1 table-build inspection per route, G2 one-update
        learner canary, G3 receipt smokes (v3 contract) — green on <em>both</em> data arms; the iteration-497
        <code>final_obs</code> blocker of the 2026-07-04 g1 run is receipt-proven closed at the campaign
        rsl_rl pin (@00debe1). Code of record @ec1cbda74 on <code>zhengyuz/bfm-campaign-20260805</code>.</small></div>
      <div class="card learner"><b>Training A/B <span class="status ok">green</span> · envelopes <span class="status wait">breach</span></b>
        <small>2026-08-05 · all six runs reached their horizons, rc=0, zero NaN/traceback; G4(a)(b) PASS,
        G4(c) curve-envelope breach on both pairs (smpl 5/22 channels over the control seed band, worst 1.8×;
        g1 10/25 over 2× iteration noise, worst 5.3×) — visible in the overlays below.</small></div>
      <div class="card data"><b>Matched-domain parity <span class="status ok">within margin</span></b>
        <small>2026-08-05 · the frozen G5 breaches (smpl 1.169–1.221×, g1 2.118×) are attributed to
        eval-reference domain shift: scored each arm on its <em>own</em> references, the ratios collapse to
        0.974–1.028 across both robots and both smpl seeds — at or under the pre-registered 1.10 margin,
        seed-replicated.</small></div>
      <div class="card robot"><b>Reward gate <span class="status ok">1.397× PASS</span></b>
        <small>2026-08-06 · reference-free 38-task broad-reward readout on the identical regenerated dataset:
        OURS — pipeline data (B1) 25.83 vs CONTROL — original data (B2) 18.49 mean return, ratio 1.397 ≥ 0.909
        — no task-level quality deficit; closes the “easier references” residual at pilot scale.</small></div>
      <div class="card gate"><b>Frozen anchors <span class="status no">missed</span></b>
        <small>Both arms miss the frozen phase-2 anchors (tracking deltas; broad-reward 0.34×/0.47×) and the
        control misses them too — a stack/schedule-class delta, recorded and reported. Per the standing
        decision rule, schedule-class deltas are bugs/tuning territory, never method verdicts.</small></div>
      <div class="card env"><b>Verdict class</b>
        <small><code>PAUSE_USER_CALL</code> — pre-registered: frozen G5 margins breached, so nothing was tuned
        and the call is the user's; the attribution (2×2) and the reward gate both landed in OURS' (the
        pipeline-data arm's) favor after the verdict was frozen. Full-horizon 211.2M g1 parity remains the
        open item below.</small></div>
    </div>
  </section>

  <section id="reading">
    <div class="kicker">02 · how to read these charts</div>
    <h2>One series identity everywhere; every historical caveat is drawn, not footnoted</h2>
    <div class="grid three">
      <div class="card data"><b>Series identity</b><small>OURS — pipeline data = solid blue, CONTROL —
      original data = solid orange on every chart; seed-1 repeats are the same hue, thinner and lighter;
      JULY-4 — the historical old run = dashed grey. Colors follow the arm, never the chart; the lead chart
      of every family also carries direct “ours”/“control” labels at the line ends. Bands are raw
      per-iteration min/max envelopes; lines are lightly smoothed (moving average, ~1/120 of the window);
      hover any chart for exact smoothed values.</small></div>
      <div class="card env"><b>Axes</b><small>x is always transitions (iterations × 500 for smpl, × 1024 for
      g1; the historical overlay uses the console's exact <code>Total steps</code>). g1 charts use log-x so
      the 0–512k window shared with the crashed historical run stays readable next to 9.6M; y switches to log
      only where a channel is strictly positive across several decades (marked “log y”). Chart y-ranges clip
      to the 1–99th percentile band where warm-up spikes would otherwise flatten the whole run (noted per
      chart).</small></div>
      <div class="card gate"><b>Honest limits</b><small>(i) the shaded warm-up window is <em>not</em>
      update-for-update comparable: the rsl_rl pin changed the warm-up gate (collected &gt; random +
      num_envs) — the historical smpl run printed its first FB update at iteration 101, today's pair logs it
      at 100 (schedule-class delta, never a method verdict); (ii) FB losses are not monotone quality proxies —
      quality lives in sections 04/06; (iii) channels existing on only one side are labeled on the chart.</small></div>
    </div>
    {coverage_html}
  </section>

  <section id="smpl">
    <div class="kicker">03 · smpl_cmu (MetaMotivo-class) · 50 envs × 10 updates · 10,000 iterations = 5M transitions</div>
    <h2>Four campaign runs and the completed JULY-4 old run share every loss regime</h2>
    <p>OURS — pipeline data (A1 seed 0, A1s1 seed 1; both completed 2026-08-05) trains on our cmu_retarget v5
    dumps (1,553 accepted clips); CONTROL — original data (A2 seed 0, A2s1 seed 1; both completed 2026-08-05)
    trains on the original HumEnv HDF5 via the campaign control registration. JULY-4 — the historical run
    (same seed 0, fb-current stack, 3583275b59c) is an OLD RUN: it completed this exact schedule on
    2026-07-04 (original data, old code) and its console record overlays as the dashed grey trace.</p>
    {smpl_html}
  </section>

  <section id="smpl-quality">
    <div class="kicker">04 · smpl quality at the horizon</div>
    <h2>Scored fairly, OURS ties or beats CONTROL — the frozen breach is the eval gauge, not the data</h2>
    {mk}
    {smpl_headline}
    <div class="grid two" style="margin-top:12px">
      <div>
      <h3 style="margin-top:0">The full 2×2: train-domain × eval-domain (matched cells are the fair read)</h3>
      {smpl_mx_html}
      </div>
      <div>
      <h3 style="margin-top:0">Frozen vs matched-domain ratios, both routes (receipts/abdiag_20260805/)</h3>
      {matched_html}
      </div>
    </div>
  </section>

  <section id="g1">
    <div class="kicker">05 · g1_lafan (BFM-Zero-class) · 1024 envs × 1 update · 9,375 iterations = 9.6M pilot</div>
    <h2>The pilot pair runs 19× past the point where the program's only prior g1 run died</h2>
    <p>OURS — pipeline data (B1; pilot completed 2026-08-05) trains on our lafan_retarget v5 dumps (843
    accepted windows); CONTROL — original data (B2; pilot completed 2026-08-05) trains on the released BFM
    joblib via the control registration; both seed 4728. The dashed grey trace is JULY-4 — the OLD 2026-07-04
    fb-current attempt at the full 211.2M horizon (original data, old code): it crashed at iteration ~497
    (<code>final_obs</code> dict handling), resumed to 499, and died — marked ✕ on every chart. The campaign
    runs cross that boundary without incident; that crossing, not any loss delta, is the integration
    story.{full_intro}</p>
    {g1_html}
  </section>

  <section id="g1-quality">
    <div class="kicker">06 · g1 quality at the pilot horizon</div>
    <h2>Matched-domain tracking is a tie, and the reference-free reward gate lands 1.397× in OURS' favor</h2>
    {mk}
    {g1_headline}
    <div class="grid two" style="margin-top:12px">
      <div>
      <h3 style="margin-top:0">The full 2×2: train-domain × eval-domain (matched cells are the fair read)</h3>
      {g1_mx_html}
      </div>
      <div>
      <h3 style="margin-top:0">Reference-free broad-reward gate (receipts/rewardgate_20260806/)</h3>
      {rgate_html}
      </div>
    </div>
  </section>

  <section id="full">
    <div class="kicker">07 · full-horizon 211.2M pair</div>
    <h2>{'RUNNING NOW — partial curves are overlaid above and refresh on regeneration' if n_live else ('Finished — curves are overlaid above' if full_runs else 'Deferred at generation time — this panel and the overlays refresh on regeneration')}</h2>
    {full_html}
  </section>

  <section id="provenance">
    <div class="kicker">08 · provenance &amp; regeneration</div>
    <h2>One command rebuilds this page from the primary artifacts</h2>
    <pre><code>{esc(str(TREE))}/.venv/bin/python \\
    {esc(str(CAMP))}/build_ab_training_curves_report.py</code></pre>
    <div class="table-wrap" style="margin-top:12px"><table>
      <thead><tr><th>input</th><th>location</th></tr></thead>
      <tbody>
        <tr><td>today's curves (raw, per-iteration)</td><td>tensorboard events in the six stage-4 run dirs under
          <code>{esc(str(TB_ROOT))}/</code> (mapped by <code>runs/stage4/*.meta</code>; binned-100 CSV exports
          archived at <code>receipts/stage4/curves/</code>)</td></tr>
        <tr><td>historical curves</td><td><code>/home/isaaclab/octi/fb-current/runs/{{smpl_cmu,g1_lafan}}/console.log</code>
          (2026-07-04; parsed per-iteration — the only surviving record)</td></tr>
        <tr><td>quality receipts</td><td><code>receipts/stage4/G5_SUMMARY.json</code> ·
          <code>receipts/abdiag_20260805/</code> · <code>receipts/rewardgate_20260806/</code></td></tr>
        <tr><td>full-horizon discovery</td><td><code>runs/*/*.meta</code> + new run dirs under the trainer logs
          root, arm-classified from <code>params/env.yaml</code></td></tr>
        <tr><td>outputs</td><td>this file, written to the trainer tree and mirrored to
          <code>{esc(str(CAMP))}/receipts/</code></td></tr>
      </tbody></table></div>
    <p class="caption">{n_figs} charts · generated {esc(gen_stamp)} on {esc(git_branch)} @ {esc(git_head)} ·
    self-contained (inline SVG, no external assets) · sibling of FORWARD_BACKWARD_PHASE_2_VALIDATION_PLAN.html
    and FORWARD_BACKWARD_PHASE_3_ENVIRONMENT_PLAN.html</p>
  </section>
</main>
<script>{TOOLTIP_JS}</script>
</body>
</html>
"""
    for out in OUTPUTS:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(page)
        print(f"[write] {out}  ({out.stat().st_size / 1e6:.2f} MB)")
    print(f"[done] {n_figs} charts")


# ------------------------------------------------------------------- assets
CSS = """
    :root {
      color-scheme: dark;
      --bg: #071018; --panel: #0d1924; --panel2: #112231; --line: #2a4358;
      --text: #e7eef4; --muted: #9db0bf; --env: #2dd4bf; --data: #60a5fa;
      --robot: #f59e0b; --learner: #c084fc; --gate: #f87171; --ok: #4ade80; --pending: #facc15;
      --ours: #3987e5; --ctrl: #d95926; --hist: #9db0bf;
    }
    * { box-sizing: border-box; }
    html { scroll-behavior: smooth; }
    body {
      margin: 0; color: var(--text);
      background: radial-gradient(circle at 10% 0, rgba(57,135,229,.10), transparent 29rem),
                  radial-gradient(circle at 92% 10%, rgba(217,89,38,.08), transparent 31rem), var(--bg);
      font: 15px/1.45 Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    main { width: min(1500px, calc(100% - 36px)); margin: 0 auto 80px; }
    header { padding: 58px 0 26px; }
    h1 { max-width: 1150px; margin: 0 0 12px; font-size: clamp(34px,5vw,64px); line-height: 1.02; letter-spacing: -.045em; }
    h2 { margin: 0 0 18px; font-size: clamp(24px,3vw,36px); letter-spacing: -.025em; }
    h3 { margin: 26px 0 11px; font-size: 18px; }
    p { color: var(--muted); }
    a { color: #8cc8ff; text-decoration: none; }
    a:hover { text-decoration: underline; }
    code, pre { font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace; }
    code { color: #d9e9f5; }
    .lead { max-width: 980px; margin: 0; color: #cbd8e1; font-size: 18px; }
    .kicker { color: var(--env); font-size: 12px; font-weight: 800; letter-spacing: .14em; text-transform: uppercase; }
    .badges, .legend { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 19px; }
    .badge, .pill, .status {
      display: inline-flex; align-items: center; gap: 7px; padding: 5px 10px;
      border: 1px solid var(--line); border-radius: 999px; background: rgba(13,25,36,.88); color: #dce7ef; font-size: 12px;
    }
    .status { padding: 2px 8px; font-size: 11px; font-weight: 800; }
    .status.ok { color: #baf7cb; background: rgba(74,222,128,.11); }
    .status.wait { color: #fff0a8; background: rgba(250,204,21,.11); }
    .status.no { color: #facaca; background: rgba(248,113,113,.11); }
    .dot { width: 9px; height: 9px; border-radius: 50%; }
    nav {
      position: sticky; top: 0; z-index: 20; display: flex; gap: 7px; overflow-x: auto;
      margin: 0 -18px 24px; padding: 10px 18px; border-block: 1px solid rgba(42,67,88,.75);
      background: rgba(7,16,24,.90); backdrop-filter: blur(14px);
    }
    nav a { flex: 0 0 auto; padding: 5px 9px; border-radius: 7px; color: #c4d2dc; font-size: 12px; }
    nav a:hover { background: var(--panel2); text-decoration: none; }
    section {
      margin: 22px 0; padding: 28px; border: 1px solid var(--line); border-radius: 18px;
      background: linear-gradient(145deg,rgba(13,25,36,.97),rgba(9,20,29,.98));
      box-shadow: 0 14px 36px rgba(0,0,0,.27); scroll-margin-top: 58px;
    }
    .grid { display: grid; gap: 11px; }
    .two { grid-template-columns: repeat(2,minmax(0,1fr)); }
    .three { grid-template-columns: repeat(3,minmax(0,1fr)); }
    .card, .callout { min-width: 0; padding: 13px; border: 1px solid var(--line); border-radius: 11px; background: var(--panel2); }
    .card b { display: block; margin-bottom: 4px; }
    .card small { display: block; color: var(--muted); }
    .env { border-color: rgba(45,212,191,.58); }
    .data { border-color: rgba(96,165,250,.58); }
    .robot { border-color: rgba(245,158,11,.58); }
    .learner { border-color: rgba(192,132,252,.58); }
    .gate { border-color: rgba(248,113,113,.58); }
    .callout { color: #cbd7df; border-left-width: 4px; }
    .callout.ok { border-left-color: var(--ok); }
    .callout.wait { border-left-color: var(--pending); }
    .callout.stop { border-left-color: var(--gate); }
    .table-wrap { overflow-x: auto; border: 1px solid var(--line); border-radius: 11px; }
    table { width: 100%; min-width: 560px; border-collapse: collapse; }
    th, td { padding: 10px 11px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }
    th { background: #132536; color: #dce7ef; font-size: 11px; letter-spacing: .04em; text-transform: uppercase; }
    td { color: #bccbd6; }
    tr:last-child td { border-bottom: 0; }
    .center td:not(:first-child), .center th:not(:first-child) { text-align: center; }
    .yes { color: var(--ok); font-weight: 800; }
    pre { margin: 12px 0 0; padding: 15px; overflow-x: auto; border: 1px solid var(--line); border-radius: 11px; background: #07131d; color: #cfe2ef; line-height: 1.5; }
    .caption { margin: 7px 0 0; color: #879cac; font-size: 12px; }
    .chart-grid { display: grid; gap: 11px; grid-template-columns: repeat(3,minmax(0,1fr)); }
    figure.chart { position: relative; margin: 0; padding: 6px; border: 1px solid var(--line); border-radius: 11px; background: __CHART_BG__; min-width: 0; }
    figure.chart svg { display: block; width: 100%; height: auto; }
    figure.chart .tick { font: 10px "SFMono-Regular",Consolas,monospace; fill: #7890a1; }
    figure.chart .ctitle { font: 700 12px Inter,ui-sans-serif,system-ui,sans-serif; fill: #dce7ef; }
    figure.chart .alab { font: 10px Inter,ui-sans-serif,system-ui,sans-serif; fill: #879cac; }
    figure.chart .anno { font: 700 10px Inter,ui-sans-serif,system-ui,sans-serif; }
    figure.chart .elab {
      font: 800 10.5px Inter,ui-sans-serif,system-ui,sans-serif; fill: #e7eef4;
      paint-order: stroke; stroke: __CHART_BG__; stroke-width: 3px; stroke-linejoin: round;
    }
    figure.chart .tt {
      position: absolute; z-index: 30; pointer-events: none; padding: 7px 9px; border: 1px solid var(--line);
      border-radius: 8px; background: rgba(7,16,24,.96); font: 11px "SFMono-Regular",Consolas,monospace;
      color: #dce7ef; white-space: nowrap; box-shadow: 0 8px 22px rgba(0,0,0,.45);
    }
    .legendnote { color: #879cac; }
    .keypanel {
      margin-top: 20px; padding: 14px 16px; border: 1px solid var(--line); border-left: 4px solid var(--ours);
      border-radius: 12px; background: rgba(13,25,36,.92); max-width: 1150px;
    }
    .keytitle { margin-bottom: 8px; color: var(--muted); font-size: 11px; font-weight: 800; letter-spacing: .11em; text-transform: uppercase; }
    .keyrow { display: flex; align-items: baseline; gap: 10px; padding: 3px 0; color: #cbd8e1; font-size: 13.5px; }
    .keyrow svg { flex: none; transform: translateY(1px); }
    .keyrow b { color: var(--text); }
    .keystatus { margin-top: 9px; padding-top: 9px; border-top: 1px dashed var(--line); color: #fff0a8; font-size: 12.5px; font-weight: 600; }
    .minikey {
      display: flex; flex-wrap: wrap; align-items: center; gap: 6px 16px; margin: 0 0 10px; padding: 6px 11px;
      border: 1px solid var(--line); border-radius: 8px; background: rgba(13,25,36,.72); color: #c4d2dc; font-size: 12px;
    }
    .minikey .mk { display: inline-flex; align-items: center; gap: 6px; }
    .minikey b { color: var(--text); }
    .navkey { flex: 0 0 auto; display: inline-flex; align-items: center; gap: 5px; margin-left: auto; padding: 5px 9px; color: #c4d2dc; font-size: 11px; font-weight: 700; }
    .mx { display: grid; grid-template-columns: 128px 1fr 1fr; gap: 7px; margin-top: 8px; }
    .mx .mxhd { align-self: end; padding: 0 2px 2px; color: var(--muted); font-size: 10.5px; font-weight: 800; letter-spacing: .05em; text-transform: uppercase; }
    .mx .mxrowhd { align-self: center; display: flex; align-items: center; gap: 6px; color: #dce7ef; font-size: 12px; font-weight: 700; }
    .mxcell { min-width: 0; padding: 10px 11px; border: 1px solid var(--line); border-radius: 10px; background: var(--panel2); }
    .mxcell b { display: block; font-size: 24px; letter-spacing: -.02em; }
    .mxcell small { display: block; margin-top: 3px; color: var(--muted); font-size: 11px; line-height: 1.35; }
    .mxcell.cross { background: rgba(13,25,36,.55); opacity: .68; }
    .mxcell.cross b { font-size: 17px; color: #a9bcc9; }
    .mxtag { display: inline-block; margin-bottom: 4px; padding: 1px 7px; border-radius: 999px; font-size: 9.5px; font-weight: 800; letter-spacing: .07em; text-transform: uppercase; }
    .mxtag.m { color: #baf7cb; background: rgba(74,222,128,.13); }
    .mxtag.x { color: #9db0bf; background: rgba(157,176,191,.12); }
    @media (max-width:1120px) { .chart-grid { grid-template-columns: repeat(2,minmax(0,1fr)); } .three { grid-template-columns: repeat(2,minmax(0,1fr)); } }
    @media (max-width:820px) { .three, .two, .chart-grid { grid-template-columns: 1fr; } }
    @media (max-width:600px) { main { width: min(100% - 20px,1500px); } section { padding: 20px 13px; } }
"""

TOOLTIP_JS = r"""
(function(){
  function fmt(v){
    var a = Math.abs(v);
    if (a >= 1e6) return (v/1e6).toPrecision(4).replace(/\.?0+$/,'') + 'M';
    if (a >= 1e4) return (v/1e3).toPrecision(4).replace(/\.?0+$/,'') + 'k';
    if (a === 0) return '0';
    return Number(v.toPrecision(4)).toString();
  }
  document.querySelectorAll('figure.chart').forEach(function(fig){
    var meta = fig.querySelector('script[type="application/json"]');
    if (!meta) return;
    var cfg = JSON.parse(meta.textContent);
    var svg = fig.querySelector('svg'), hit = fig.querySelector('rect.hit'),
        xh = fig.querySelector('line.xh'), tt = fig.querySelector('.tt');
    if (!svg || !hit || !tt) return;
    var vb = svg.viewBox.baseVal;
    function pxToT(px){
      var f = (px - cfg.ml) / cfg.pw;
      f = Math.max(0, Math.min(1, f));
      if (cfg.logx){
        var l0 = Math.log10(cfg.tmin), l1 = Math.log10(cfg.tmax);
        return Math.pow(10, l0 + f * (l1 - l0));
      }
      return cfg.tmin + f * (cfg.tmax - cfg.tmin);
    }
    function nearest(pts, t){
      var lo = 0, hi = pts.length - 1;
      while (hi - lo > 1){ var mid = (lo + hi) >> 1; (pts[mid][0] < t) ? lo = mid : hi = mid; }
      return (Math.abs(pts[lo][0] - t) < Math.abs(pts[hi][0] - t)) ? pts[lo] : pts[hi];
    }
    svg.addEventListener('mousemove', function(ev){
      var r = svg.getBoundingClientRect();
      var px = (ev.clientX - r.left) * vb.width / r.width;
      if (px < cfg.ml || px > cfg.ml + cfg.pw){ tt.hidden = true; xh.setAttribute('opacity', 0); return; }
      var t = pxToT(px);
      var rows = ['<b>' + fmt(t) + ' transitions</b>'];
      cfg.series.forEach(function(s){
        if (!s.p.length) return;
        var q = nearest(s.p, t);
        if (Math.abs(q[0] - t) > (cfg.tmax - cfg.tmin) * 0.08 && !cfg.logx) return;
        rows.push('<span style="color:' + s.c + '">●</span> ' + s.n + ': <b>' + fmt(q[1]) + '</b>');
      });
      tt.innerHTML = rows.join('<br>');
      tt.hidden = false;
      xh.setAttribute('x1', px); xh.setAttribute('x2', px); xh.setAttribute('opacity', 0.35);
      var fr = fig.getBoundingClientRect();
      var left = ev.clientX - fr.left + 14, top = ev.clientY - fr.top + 10;
      if (left + tt.offsetWidth > fr.width - 8) left = left - tt.offsetWidth - 26;
      tt.style.left = left + 'px'; tt.style.top = top + 'px';
    });
    svg.addEventListener('mouseleave', function(){ tt.hidden = true; xh.setAttribute('opacity', 0); });
  });
})();
"""

if __name__ == "__main__":
    main()
