#!/usr/bin/env python3
"""
Plot event timings from a PETSc/SLEPc -log_view performance summary.

Usage:
    python plot_petsc_log.py logfile.txt
    python plot_petsc_log.py logfile.txt --top 20 --metric time --out timings.png
    cat logfile.txt | python plot_petsc_log.py -          # read from stdin

The script parses the per-event table (the block whose header starts with
"Event   Count   Time (sec) ...") and plots a horizontal bar chart sorted
by the chosen metric.
"""

import argparse
import re
import sys

import matplotlib.pyplot as plt


# Each data row looks like:
#   MatMult  89 1.1 2.0782e-02 3.3 5.56e+07 1.5 1.8e+02 8.6e+02 0.0e+00  3 87 16 11  0   7 87 20 13  0  9555
#
# Layout after the event name:
#   Count Ratio | Time Ratio | Flop Ratio | Mess AvgLen Reduct |
#   %T %F %M %L %R (global) | %T %F %M %L %R (stage) | Mflop/s
NUM = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
ROW_RE = re.compile(
    r"^\s*(?P<name>\S+)\s+"
    r"(?P<count>\d+)\s+(?P<count_ratio>" + NUM + r")\s+"
    r"(?P<time>" + NUM + r")\s+(?P<time_ratio>" + NUM + r")\s+"
    r"(?P<flop>" + NUM + r")\s+(?P<flop_ratio>" + NUM + r")\s+"
    r"(?P<mess>" + NUM + r")\s+(?P<avglen>" + NUM + r")\s+(?P<reduct>" + NUM + r")\s+"
    r"(?P<gpt>\d+)\s+(?P<gpf>\d+)\s+(?P<gpm>\d+)\s+(?P<gpl>\d+)\s+(?P<gpr>\d+)\s+"
    r"(?P<spt>\d+)\s+(?P<spf>\d+)\s+(?P<spm>\d+)\s+(?P<spl>\d+)\s+(?P<spr>\d+)\s+"
    r"(?P<mflops>\d+)\s*$"
)

# Categorise events by name prefix for colouring.
CATEGORIES = [
    (
        "setup",
        (
            "PCSetUp",
            "PCHPDDM",
            "MatLUFactor",
            "MatCholFctr",
            "MatICCFactor",
            "EPSSetUp",
            "STSetUp",
            "KSPSetUp",
            "SFSetUp",
            "SFSetGraph",
        ),
    ),
    (
        "solve",
        (
            "EPSSolve",
            "KSPSolve",
            "MatSolve",
            "STApply",
            "STMatSolve",
            "PCApply",
            "BVMatMultVec",
            "DSSolve",
        ),
    ),
    ("mat", ("Mat",)),
    ("vec", ("Vec", "BV", "SF")),
]
CAT_COLORS = {
    "setup": "#534AB7",
    "solve": "#1D9E75",
    "mat": "#D85A30",
    "vec": "#3B82C4",
    "other": "#888780",
}

# Map a metric name to the parsed field and an axis label.
METRICS = {
    "time": ("time", "Max time (s)"),
    "flop": ("flop", "Flop (max)"),
    "mflops": ("mflops", "Mflop/s"),
    "pt": ("gpt", "% of total time"),
    "ratio": ("time_ratio", "Time imbalance (max/min)"),
    "count": ("count", "Call count"),
}


def categorise(name):
    for cat, prefixes in CATEGORIES:
        if name.startswith(prefixes):
            return cat
    return "other"


def parse_log(text):
    """Return a list of dicts, one per parsed event row."""
    events = []
    for line in text.splitlines():
        if line.startswith("---") or not line.strip():
            continue
        m = ROW_RE.match(line)
        if not m:
            continue
        d = m.groupdict()
        rec = {"name": d["name"]}
        for k, v in d.items():
            if k == "name":
                continue
            rec[k] = float(v)
        rec["category"] = categorise(rec["name"])
        events.append(rec)
    return events


def plot(events, metric, top, out):
    field, axis_label = METRICS[metric]
    events = [e for e in events if e[field] > 0]
    events.sort(key=lambda e: e[field], reverse=True)
    events = events[:top]
    events.reverse()  # largest at top of horizontal bar chart

    names = [e["name"] for e in events]
    values = [e[field] for e in events]
    colors = [CAT_COLORS[e["category"]] for e in events]

    fig, ax = plt.subplots(figsize=(9, max(3, 0.35 * len(events) + 1)))
    ax.barh(names, values, color=colors)
    ax.set_xlabel(axis_label)
    ax.set_title(f"PETSc/SLEPc events by {metric} (top {len(events)})")
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(axis="x", color="0.85", linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    seen = {}
    for e in events:
        seen[e["category"]] = CAT_COLORS[e["category"]]
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in seen.values()]
    ax.legend(handles, seen.keys(), loc="lower right", frameon=False, fontsize=9)

    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=150)
        print(f"Saved {out}")
    else:
        plt.show()


def main():
    p = argparse.ArgumentParser(description="Plot PETSc/SLEPc -log_view event timings")
    p.add_argument("logfile", help="path to the log file, or '-' for stdin")
    p.add_argument(
        "--metric",
        default="time",
        choices=METRICS,
        help="which column to plot (default: time)",
    )
    p.add_argument("--top", type=int, default=20, help="number of events to show")
    p.add_argument(
        "--out", default=None, help="save to this image file instead of showing"
    )
    args = p.parse_args()

    text = sys.stdin.read() if args.logfile == "-" else open(args.logfile).read()
    events = parse_log(text)
    if not events:
        sys.exit("No event rows parsed - check the log format.")
    print(f"Parsed {len(events)} events.")
    plot(events, args.metric, args.top, args.out)


if __name__ == "__main__":
    main()
