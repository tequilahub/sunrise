"""Convert qpic files into images, with ``qpic`` or matplotlib."""

import os
import re
import shutil
import subprocess
from math import pi


def qpic_to_pdf(filename, filepath=None):
    """Run ``qpic`` to convert ``<filename>.qpic`` into a PDF.

    Falls back to ``qpic_to_matplotlib`` when ``qpic`` (or the ``pdflatex`` it calls) is missing
    or fails.

    Parameters
    ----------
    filename : str
        Name of the qpic file without the ``.qpic`` extension.
    filepath : str, optional
        Directory containing the file. Default is the current working directory.
    """
    if shutil.which("qpic") is None or subprocess.call(
        ["qpic", f"{filename}.qpic", "-f", "pdf"], cwd=filepath or os.getcwd()
    ):
        qpic_to_matplotlib(filename, filepath, "pdf")


def qpic_to_png(filename, filepath=None):
    """Run ``qpic`` to convert ``<filename>.qpic`` into a PNG.

    Falls back to ``qpic_to_matplotlib`` when ``qpic`` (or the ``pdflatex`` it calls) is missing
    or fails.

    Parameters
    ----------
    filename : str
        Name of the qpic file without the ``.qpic`` extension.
    filepath : str, optional
        Directory containing the file. Default is the current working directory.
    """
    if shutil.which("qpic") is None or subprocess.call(
        ["qpic", f"{filename}.qpic", "-f", "png"], cwd=filepath or os.getcwd()
    ):
        qpic_to_matplotlib(filename, filepath, "png")


def qpic_to_matplotlib(filename, filepath=None, ftype="png"):
    """Draw ``<filename>.qpic`` with matplotlib, without ``qpic`` or ``pdflatex``.

    Only the qpic subset written by ``FCircuit.export_qpic`` is understood.

    Parameters
    ----------
    filename : str
        Name of the qpic file without the ``.qpic`` extension.
    filepath : str, optional
        Directory containing the file. Default is the current working directory.
    ftype : str, optional
        Output format passed to matplotlib, e.g. ``"png"`` or ``"pdf"``.
    """
    from matplotlib.colors import to_rgb
    from matplotlib.figure import Figure
    from matplotlib.patches import Circle, Rectangle, RegularPolygon

    path = os.path.join(filepath or "", filename)
    named, wires, last_x, slashed = {}, {}, {}, set()

    def rgb(color):
        """xcolor mixing: 'a!p!b' is p% of a and the rest b, chained left to right."""
        parts = color.split("!")
        out = named.get(parts[0]) or to_rgb(parts[0])
        for p, other in zip(parts[1::2], parts[2::2]):
            o, t = named.get(other) or to_rgb(other), int(p) / 100
            out = tuple(t * a + (1 - t) * b for a, b in zip(out, o))
        return out

    fig = Figure()
    ax = fig.add_subplot(aspect="equal")
    ax.set_axis_off()
    with open(f"{path}.qpic") as file:
        for line in file:
            tokens = line.split()
            if not tokens or tokens[-1] == "TOUCH":
                continue
            if tokens[-1] == "/":  # "a0 /": a slash on the wire, as for a bundle
                slashed.update(int(t[1:]) for t in tokens[:-1])
                continue
            if tokens[0] == "COLOR":
                named[tokens[1]] = tuple(float(t) for t in tokens[2:5])
                continue
            if "W" in tokens:
                wires[int(tokens[1][1:])] = (" ".join(tokens[3:]), tokens[0][6:])
                continue
            # "a0 P:fill=c:shape=6": a shape per wire; "a0 a1 G G": a labelled box
            places = re.findall(r"a(\d+) P:fill=([^:]+):shape=(-?\d+)", line)
            ws = [int(w) for w, _, _ in places]
            ws = ws or [int(t[1:]) for t in tokens if re.fullmatch(r"a\d+", t)]
            # first column free on every wire the gate crosses, as qpic does
            span = range(min(ws), max(ws) + 1)
            x = max(last_x.get(w, 0) for w in span) + 1
            last_x.update(dict.fromkeys(span, x))
            ax.plot([x, x], [-min(ws), -max(ws)], color="black", lw=1, zorder=1)
            if not places:
                box = Rectangle((x - 0.3, -max(ws) - 0.3), 0.6, len(span) - 0.4)
                box.set(facecolor="white", edgecolor="black", zorder=2)
                ax.add_patch(box)
                y = -(min(ws) + max(ws)) / 2
                ax.text(x, y, tokens[-1], ha="center", va="center", zorder=3)
            for w, fill, shape in places:
                n, xy = int(shape), (x, -int(w))
                if n == 2:
                    patch = Circle(xy, 0.25)
                else:  # negative n: polygon with its tip at the bottom
                    patch = RegularPolygon(xy, abs(n), radius=0.28)
                    patch.orientation = pi * (n < 0)
                patch.set(facecolor=rgb(fill), edgecolor="black", zorder=2)
                ax.add_patch(patch)

    n_x = max(last_x.values(), default=0) + 1
    all_wires = sorted(set(wires) | set(last_x))
    for w in all_wires:
        label, color = wires.get(w, (str(w), "black"))
        ax.plot([0, n_x], [-w, -w], color=rgb(color), lw=1, zorder=0)
        ax.text(-0.15, -w, label, ha="right", va="center", color=rgb(color))
        if w in slashed:
            ax.plot([0.25, 0.45], [-w - 0.15, -w + 0.15], color=rgb(color), lw=1)
    ax.set(xlim=(-0.8, n_x), ylim=(-max(all_wires) - 0.7, 0.7))
    fig.set_size_inches(0.6 * n_x + 1, 0.5 * len(all_wires) + 0.5)
    fig.savefig(f"{path}.{ftype}", bbox_inches="tight", dpi=200)
