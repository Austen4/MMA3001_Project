"""
Shared figure style and plots used by the pipeline scripts.

Colours follow one fixed categorical order so the same slot means the same thing across
figures; observed data is always drawn in ink (black), never a series colour.
"""
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import config, evaluation  # noqa: E402

INK, INK_MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e6e5e0", "#fcfcfb"
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]


def style_axes(ax):
    """Apply the shared axis style: surface background, light y-grid, no spines."""
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.tick_params(colors=INK_MUTED, labelsize=8, length=0)
    for s in ax.spines.values():
        s.set_visible(False)


def prediction_table(test, predictions):
    """Observed and predicted target for every hold-out row.

    Parameters
    ----------
    test : pandas.DataFrame
        Hold-out rows (must contain ``time_utc`` and the target).
    predictions : dict
        ``model name -> array`` of predictions aligned with ``test``.

    Returns
    -------
    pandas.DataFrame
        ``time_utc``, ``time_local`` (Sydney), ``observed`` and one column per model.
    """
    out = pd.DataFrame({"time_utc": test["time_utc"].to_numpy(),
                        "time_local": test["time_utc"].dt.tz_convert(config.TIMEZONE).to_numpy(),
                        "observed": test[config.TARGET].to_numpy()})
    for name, pred in predictions.items():
        if len(pred) != len(test):
            raise ValueError(f"predictions for {name!r} do not match the test rows")
        out[name] = pred
    return out


def plot_test_period(test, predictions, path, title, days=14, metric_label="hold-out"):
    """Observed vs predicted target over the first ``days`` of the hold-out, one panel per model.

    Each panel title carries the model's MAE and R² over *all* rows of ``test``, so the
    plotted window is illustrative and the numbers are the evaluation. All panels share a
    y-axis capped at 1.15 x the observed maximum in the window; predictions above the cap
    are labelled with their value so one extreme model does not flatten the others.

    Parameters
    ----------
    test : pandas.DataFrame
        Hold-out rows, time-ordered.
    predictions : dict
        ``model name -> array`` aligned with ``test``; panels follow this order and the
        colours follow :data:`CATEGORICAL`.
    path : str or Path
        Output PNG.
    title : str
        Figure caption.
    days : float
        Length of the plotted window.
    metric_label : str
        What the rows in ``test`` are, shown before the metrics in each panel title
        (e.g. ``"in-sample"`` when the rows were used for training).
    """
    if not predictions:
        raise ValueError("no predictions to plot")
    sel = (test["time_utc"] < test["time_utc"].iloc[0] + pd.Timedelta(days=days)).to_numpy()
    t = test["time_utc"][sel].dt.tz_convert(config.TIMEZONE).dt.tz_localize(None)
    observed = test[config.TARGET].to_numpy()
    top = 1.15 * observed[sel].max()
    fig, axes = plt.subplots(len(predictions), 1, figsize=(11, 2.1 * len(predictions) + 0.8),
                             sharex=True, sharey=True, squeeze=False, facecolor=SURFACE)
    for ax, (name, pred), color in zip(axes[:, 0], predictions.items(), CATEGORICAL):
        style_axes(ax)
        m = evaluation.regression_metrics(observed, pred)
        shown = np.asarray(pred)[sel]
        ax.plot(t, observed[sel], color=INK, lw=1.2)
        ax.plot(t, np.minimum(shown, top), color=color, lw=2)
        above = np.flatnonzero(shown > top)
        for run in np.split(above, np.flatnonzero(np.diff(above) > 1) + 1) if above.size else []:
            i = run[np.argmax(shown[run])]       # label each off-scale excursion once, at its peak
            ax.annotate(f"{shown[i]:.0f}", (t.iloc[i], top), textcoords="offset points", xytext=(4, -10),
                        fontsize=8, color=INK)
        ax.set_title(f"{name}  ·  {metric_label} MAE {m['mae']:.2f}, R² {m['r2']:.2f}",
                     loc="left", fontsize=10, color=INK, fontweight="bold")
    axes[0, 0].set_ylim(0, top)
    last = axes[-1, 0]
    last.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    last.xaxis.set_major_formatter(mdates.DateFormatter("%a %d %b"))
    last.set_xlabel("Sydney local time", fontsize=9, color=INK_MUTED)
    caption = "\n".join(textwrap.wrap(title, 150))
    fig.text(0.01, 0.99, caption, fontsize=9, color=INK_MUTED, va="top")
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.025 * (caption.count("\n") + 1.2)))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
