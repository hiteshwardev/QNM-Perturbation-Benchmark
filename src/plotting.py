"""Single-panel figures at a fixed physical size, with all text kept outside the plot box.

Every figure is one set of axes on an 18 cm wide canvas (double-column width). After
plotting, the text extents are measured on the rendered canvas and the plot box is shrunk
until axis labels, tick labels and an optional legend or colour bar fit inside the canvas,
which keeps its physical size. Before a figure is written, its text is checked for overlaps,
for intrusion into the plot box and for clipping at the canvas edge; any such defect raises
an error instead of producing a faulty figure. Type sizes are 10 pt for axis labels and 9 pt
for tick labels and legends at the printed size.
"""
import io
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402

CM = 1.0 / 2.54
PNG_DPI = 600
WIDTH_CM, HEIGHT_CM = 18.0, 10.5
LEGEND_COLUMN_CM = 5.2
COLORBAR_COLUMN_CM = 2.6

# Okabe-Ito colour-blind-safe palette.
COLORS = {"black": "#000000", "orange": "#E69F00", "skyblue": "#56B4E9", "green": "#009E73",
          "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7",
          "grey": "#777777"}

STYLE = {
    "font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
    "font.size": 9, "axes.labelsize": 10, "axes.titlesize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "axes.linewidth": 0.8, "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "lines.linewidth": 1.5, "lines.markersize": 4.0,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 4, "ytick.major.size": 4, "xtick.minor.size": 2, "ytick.minor.size": 2,
    "legend.frameon": True, "legend.framealpha": 1.0, "legend.edgecolor": "0.6",
    "legend.borderpad": 0.6, "legend.labelspacing": 0.6, "legend.handlelength": 2.4,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "pdf.fonttype": 42, "ps.fonttype": 42,
}


def new_figure(width_cm=WIDTH_CM, height_cm=HEIGHT_CM, legend=False, colorbar=False):
    """Figure with one axes; legend or colorbar reserve a column on the right."""
    if legend and colorbar:
        raise ValueError("reserve either a legend column or a colour-bar column")
    plt.rcParams.update(STYLE)
    right = LEGEND_COLUMN_CM if legend else COLORBAR_COLUMN_CM if colorbar else 0.6
    fig = plt.figure(figsize=(width_cm * CM, height_cm * CM))
    left, bottom, top = 2.0, 1.5, 0.5
    ax = fig.add_axes([left / width_cm, bottom / height_cm,
                       (width_cm - left - right) / width_cm,
                       (height_cm - bottom - top) / height_cm])
    fig._layout = {"width_cm": width_cm, "height_cm": height_cm, "legend": legend,
                   "colorbar": colorbar}
    return fig, ax


def label(ax, xlabel=None, ylabel=None, pad_frac=0.04):
    """Axis labels, plus a small margin between the data and the axes frame."""
    if xlabel:
        ax.set_xlabel(xlabel, labelpad=6)
    if ylabel:
        ax.set_ylabel(ylabel, labelpad=8)
    if pad_frac:
        for scale, get, set_ in ((ax.get_xscale(), ax.get_xlim, ax.set_xlim),
                                 (ax.get_yscale(), ax.get_ylim, ax.set_ylim)):
            lo, hi = get()
            if scale == "log" and lo > 0:
                factor = (hi / lo) ** pad_frac
                set_(lo / factor, hi * factor)
            elif scale == "linear":
                set_(lo - pad_frac * (hi - lo), hi + pad_frac * (hi - lo))
    return ax


def log_ticks(ax, axis, values, labels=None):
    """Labelled major ticks at the given values on a logarithmic axis."""
    from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter

    target = ax.xaxis if axis == "x" else ax.yaxis
    target.set_major_locator(FixedLocator(values))
    target.set_major_formatter(FixedFormatter(labels or [f"{v:g}" for v in values]))
    target.set_minor_formatter(NullFormatter())


def legend_outside(fig, ax, **kwargs):
    """Legend in the reserved right-hand column."""
    if not fig._layout["legend"]:
        raise ValueError("create the figure with legend=True")
    pos = ax.get_position()
    leg = fig.legend(*ax.get_legend_handles_labels(), loc="center left",
                     bbox_to_anchor=(pos.x1 + 0.3 / fig._layout["width_cm"], pos.y0 + pos.height / 2),
                     borderaxespad=0.0, **kwargs)
    leg.set_in_layout(False)
    return leg


def colorbar_outside(fig, ax, mappable, label_text=None):
    """Colour bar in the reserved right-hand column, as tall as the plot box."""
    if not fig._layout["colorbar"]:
        raise ValueError("create the figure with colorbar=True")
    w = fig._layout["width_cm"]
    pos = ax.get_position()
    cax = fig.add_axes([pos.x1 + 0.35 / w, pos.y0, 0.45 / w, pos.height])
    cbar = fig.colorbar(mappable, cax=cax)
    if label_text:
        cbar.set_label(label_text)
    return cbar


def _ticklabels(ax, axis):
    """Tick labels whose ticks lie within the current axis limits."""
    if axis == "x":
        locs, labels, lim = ax.get_xticks(), ax.get_xticklabels(), ax.get_xlim()
    else:
        locs, labels, lim = ax.get_yticks(), ax.get_yticklabels(), ax.get_ylim()
    lo, hi = min(lim), max(lim)
    return [lab for loc, lab in zip(locs, labels)
            if lo - 1e-12 <= loc <= hi + 1e-12 and lab.get_text().strip() and lab.get_visible()]


def _fit_margins(fig, ax, max_iter=6):
    """Shrink the plot box until every text element fits on the fixed-size canvas."""
    w, h = fig._layout["width_cm"], fig._layout["height_cm"]
    extra = [a for a in fig.axes if a is not ax]
    for _ in range(max_iter):
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        renderer = canvas.get_renderer()
        pos = ax.get_position()
        x0, x1, y0, y1 = pos.x0 * w, pos.x1 * w, pos.y0 * h, pos.y1 * h

        def extent(artist, attr):
            return getattr(artist.get_window_extent(renderer=renderer), attr) / fig.dpi / CM

        left = max([0.3] + [x0 - extent(t, "x0") + 0.15 for t in _ticklabels(ax, "y")]
                   + ([x0 - extent(ax.yaxis.label, "x0") + 0.2] if ax.yaxis.label.get_text() else []))
        bottom = max([0.3] + [y0 - extent(t, "y0") + 0.15 for t in _ticklabels(ax, "x")]
                     + ([y0 - extent(ax.xaxis.label, "y0") + 0.2] if ax.xaxis.label.get_text() else []))
        top = max([0.35] + [extent(t, "y1") - y1 + 0.15 for t in _ticklabels(ax, "y")])
        right = max([0.45] + [extent(t, "x1") - x1 + 0.15 for t in _ticklabels(ax, "x")])
        for leg in fig.legends:
            right = max(right, extent(leg, "x1") - x1 + 0.3)
        for cax in extra:
            items = _ticklabels(cax, "y") + ([cax.yaxis.label] if cax.yaxis.label.get_text() else [])
            right = max([right, (cax.get_position().x1 * w - x1) + 0.3]
                        + [extent(t, "x1") - x1 + 0.3 for t in items])
        new = (left, bottom, w - left - right, h - bottom - top)
        if new[2] < 0.35 * w or new[3] < 0.35 * h:
            raise ValueError("text does not fit on the canvas; enlarge the figure or shorten labels")
        if max(abs(new[0] - x0), abs(new[1] - y0), abs(new[0] + new[2] - x1),
               abs(new[1] + new[3] - y1)) < 0.02:
            break
        ax.set_position([new[0] / w, new[1] / h, new[2] / w, new[3] / h])
        for cax in extra:
            cpos = cax.get_position()
            cax.set_position([(new[0] + new[2] + 0.35) / w, new[1] / h, cpos.width, new[3] / h])
        for leg in fig.legends:
            leg.set_bbox_to_anchor(((new[0] + new[2] + 0.3) / w, (new[1] + new[3] / 2) / h))


def _text_boxes(fig, renderer):
    boxes = []

    def add(name, artist):
        if hasattr(artist, "get_text") and not str(artist.get_text()).strip():
            return
        bb = artist.get_window_extent(renderer=renderer)
        if bb.width > 0 and bb.height > 0:
            boxes.append((name, bb))

    for i, ax in enumerate(fig.axes):
        add(f"axes{i} title", ax.title)
        add(f"axes{i} x label", ax.xaxis.label)
        add(f"axes{i} y label", ax.yaxis.label)
        for axis in ("x", "y"):
            for j, t in enumerate(_ticklabels(ax, axis)):
                add(f"axes{i} {axis} tick label {j}", t)
        for j, t in enumerate(ax.texts):
            add(f"axes{i} text {j}", t)
        if ax.get_legend() is not None:
            add(f"axes{i} legend", ax.get_legend())
    for j, t in enumerate(fig.texts):
        add(f"figure text {j}", t)
    for j, leg in enumerate(fig.legends):
        add(f"figure legend {j}", leg)
    return boxes


def _overlap_area(b1, b2, min_side=1.0, min_area=4.0):
    dx = min(b1.x1, b2.x1) - max(b1.x0, b2.x0)
    dy = min(b1.y1, b2.y1) - max(b1.y0, b2.y0)
    if dx <= min_side or dy <= min_side or dx * dy <= min_area:
        return 0.0
    return dx * dy


def layout_problems(fig):
    """Overlaps between text elements, text inside a plot box, and text clipped at the edge."""
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    renderer = canvas.get_renderer()
    boxes = _text_boxes(fig, renderer)
    problems = []
    for i, ax in enumerate(fig.axes):
        frame = ax.patch.get_window_extent(renderer=renderer)
        own = (f"axes{i} x label", f"axes{i} y label", f"axes{i} x tick", f"axes{i} y tick")
        for name, bb in boxes:
            if name.startswith(own):
                continue
            if _overlap_area(frame, bb) > 0.02 * bb.width * bb.height:
                problems.append(f"{name} lies inside the plot box of axes{i}")
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if _overlap_area(boxes[i][1], boxes[j][1]) > 0:
                problems.append(f"{boxes[i][0]} overlaps {boxes[j][0]}")
    w, h = fig.bbox.width, fig.bbox.height
    for name, bb in boxes:
        if bb.x0 < -1 or bb.y0 < -1 or bb.x1 > w + 1 or bb.y1 > h + 1:
            problems.append(f"{name} is clipped at the figure edge")
    return problems


def save(fig, path_stem):
    """Fit the plot box, check the layout, and write PDF and 600-dpi PNG files."""
    _fit_margins(fig, fig.axes[0])
    problems = layout_problems(fig)
    if problems:
        plt.close(fig)
        raise ValueError(f"layout check failed for {path_stem}: " + "; ".join(problems))
    path = pathlib.Path(path_stem)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"), metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(path.with_suffix(".png"), dpi=PNG_DPI, metadata={"Software": None})
    plt.close(fig)
    return path.with_suffix(".png")


def preview(png_path, width_px=1300):
    """Downscaled copy of a saved PNG for display in the notebook."""
    from IPython.display import Image
    from PIL import Image as PILImage

    with PILImage.open(png_path) as img:
        scale = width_px / img.width
        small = img.convert("RGB").resize((width_px, round(img.height * scale)), PILImage.LANCZOS)
    buffer = io.BytesIO()
    small.save(buffer, format="PNG", optimize=True)
    return Image(data=buffer.getvalue(), width=width_px // 2)
