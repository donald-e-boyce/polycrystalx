"""Convergence scatterplots from HDF5 output of run_convergence.py.

Each point is one grain from one suite (microstructure).  Suites are
distinguished by colour.  Two figures are produced — one for temperature,
one for flux — each with a 2 × 2 grid (or 2 × 3 when point-error data is
present):

    rows:    convergence rate (m)  |  coefficient (c)
    cols:    continuous-grain error  |  grain-average error  [|  point error]

Within each row the y-axis limits are shared across all columns by default,
so the error types can be compared directly.

With ``--ridge`` two additional figures are produced (one per field) showing
the convergence fits at the grain-interface (ridge) evaluation points:

    rows:    convergence rate (m)  |  coefficient (c)
    cols:    one column per Voronoi suite

The x-axis in each panel is the inter-seed distance of the Voronoi ridge.

Usage
-----
    python plot_convergence.py <data.h5> [--show] [--ridge]

Figures are saved alongside the HDF5 file as::

    <stem>-temperature.png
    <stem>-flux.png
    <stem>-ridge-temperature.png     # only with --ridge
    <stem>-ridge-flux.png            # only with --ridge
"""
import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

# ------------------------------------------------------------------ #
# Data loading                                                         #
# ------------------------------------------------------------------ #

def _compute_r(errors, h, rates, log_intercepts):
    """Compute the correlation coefficient R of the log-log fit for each grain/point.

    Parameters
    ----------
    errors : ndarray, shape (njobs, ngrains)
    h : ndarray, shape (njobs,)       — mesh-size parameters (coarse meshes only)
    rates : ndarray, shape (ngrains,) — fitted slopes
    log_intercepts : ndarray, shape (ngrains,) — fitted intercepts in log10 space

    Returns
    -------
    ndarray, shape (ngrains,)  — values in [0, 1]
    """
    x = np.log10(h)
    ngrains = errors.shape[1]
    r = np.zeros(ngrains)
    for g in range(ngrains):
        y = np.log10(errors[:, g])
        y_hat = rates[g] * x + log_intercepts[g]
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
        r[g] = np.sqrt(max(0.0, r2))
    return r


def load_suite(hf, suite_name):
    """Load per-grain data for one suite from an open HDF5 file.

    Parameters
    ----------
    hf : h5py.File
    suite_name : str

    Returns
    -------
    dict with keys:
        volume        ndarray (ngrains,)   — finest-mesh grain volumes (%)
        grain_avg     dict  field → {rates, intercepts, r2}
        continuous    dict  field → {rates, intercepts, r2}
        point_errors  dict  field → {rates, intercepts, r2}   (if present)
    """
    grp = hf[suite_name]

    # Use the finest mesh (last row) and express as percentage of total volume.
    volume = grp["volume"][-1, :]
    volume = volume / volume.sum() * 100.0

    # h[:-1] are the coarse-mesh sizes used for fitting (finest is the reference).
    h_coarse = grp["h"][:-1]

    def load_params(error_type, field):
        fg = grp[error_type][field]
        rates         = fg["rates"][:]
        log_intercepts = fg["intercepts"][:]      # stored in log10 space
        errors        = fg["errors"][:]           # (njobs-1, ngrains)
        return {
            "rates":      rates,
            "intercepts": 10.0 ** log_intercepts,
            "r":          _compute_r(errors, h_coarse, rates, log_intercepts),
        }

    result = {
        "volume":     volume,
        "grain_avg":  {f: load_params("grain_avg",       f) for f in ("temperature", "flux")},
        "continuous": {f: load_params("continuous_grain", f) for f in ("temperature", "flux")},
    }

    if "point_errors" in grp:
        result["point_errors"] = {
            f: load_params("point_errors", f) for f in ("temperature", "flux")
        }

    if "ridge_errors" in grp:
        # Inter-seed distance: the 45 % and 55 % points are 10 % of the
        # inter-seed distance apart, so multiply their separation by 10.
        ridge_pts = grp["ridge_errors"]["ridge_points"][:]   # (2*n_ridges, 3)
        inter_seed_dists = 10.0 * np.linalg.norm(
            ridge_pts[1::2] - ridge_pts[0::2], axis=1       # (n_ridges,)
        )
        # Both points of a pair share the same x-coordinate.
        x_vals = np.repeat(inter_seed_dists, 2)             # (2*n_ridges,)
        result["ridge_errors"] = {
            "x": x_vals,
            **{f: load_params("ridge_errors", f) for f in ("temperature", "flux")},
        }

    return result


def load_all_suites(h5_path):
    """Load data for every suite in the HDF5 file.

    Returns
    -------
    list of (suite_name, suite_dict) pairs in file order.
    """
    with h5py.File(h5_path, "r") as hf:
        return [(name, load_suite(hf, name)) for name in hf.keys()]


# ------------------------------------------------------------------ #
# Plotting helpers                                                      #
# ------------------------------------------------------------------ #

_CMAP = plt.cm.gray_r   # high shade value → dark/black; low → light grey

# Labels for the colorbar depending on the shading choice.
_SHADE_LABEL = {"R": "R", "R2": "$R^2$"}


def _shade_values(r, shade):
    """Return the colour values to use for the scatter plot.

    Parameters
    ----------
    r : ndarray
        Correlation coefficient R for each point (values in [0, 1]).
    shade : "R" | "R2"
        Which quantity to map to colour.

    Returns
    -------
    ndarray — same shape as *r*, values in [0, 1].
    """
    return r ** 2 if shade == "R2" else r


def _scatter_ax(ax, suites, error_type, field, param, shade="R2"):
    """Populate a single Axes with one scatter panel.

    Points are coloured by R or R² according to *shade*.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    suites : list of (name, dict)
    error_type : "grain_avg" | "continuous" | "point_errors"
    field : "temperature" | "flux"
    param : "rates" | "intercepts"
    shade : "R" | "R2"
        Quantity used for point colour.  Default ``"R2"``.
    """
    for _, data in suites:
        volume = data["volume"]
        values = data[error_type][field][param]
        c      = _shade_values(data[error_type][field]["r"], shade)
        ax.scatter(volume, values, c=c, cmap=_CMAP, vmin=0, vmax=1,
                   s=30, alpha=0.8)


def _limits(vals, param):
    """Return (ymin, ymax) with appropriate margins for *param*."""
    lo, hi = vals.min(), vals.max()
    if param == "intercepts":
        margin = 0.05 * hi
        return max(0.0, lo - margin), hi + margin
    else:
        margin = 0.05 * (hi - lo) if hi != lo else 0.1
        return lo - margin, hi + margin


def _param_limits(suites, field, param):
    """Return (ymin, ymax) spanning all suites and all error types."""
    arrays = (
        [data["grain_avg"][field][param]  for _, data in suites]
        + [data["continuous"][field][param] for _, data in suites]
        + [data["point_errors"][field][param]
           for _, data in suites if "point_errors" in data]
    )
    return _limits(np.concatenate(arrays), param)


def _panel_limits(active, error_type, field, param):
    """Return (ymin, ymax) for a single panel (independent scaling)."""
    vals = np.concatenate([
        data[error_type][field][param] for _, data in active
    ])
    return _limits(vals, param)


_PARAM_LABELS = {
    "rates":      "convergence rate  m",
    "intercepts": "coefficient  c",
}

_ERROR_TITLES = {
    "grain_avg":    "grain-average error",
    "continuous":   "continuous-grain error",
    "point_errors": "point error at seed",
}

_FIELD_TITLES = {
    "temperature": "Temperature",
    "flux":        "Flux",
}


def make_figure(suites, field, shade="R2"):
    """Build the 2 × 2 (or 2 × 3) figure for one field.

    A third column showing point errors at the Voronoi seed points is added
    automatically when that data is present in the loaded suites.  Each grain
    is coloured by R or R² (see *shade*); a shared colorbar is added to the
    right of the figure.

    The coefficient (intercept) row always uses a log10 y-axis with
    per-panel limits.

    Parameters
    ----------
    suites : list of (name, dict)
    field : "temperature" | "flux"
    shade : "R" | "R2"
        Quantity used for point colour.  Default ``"R2"``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    params = ("rates", "intercepts")
    has_point_errors = any("point_errors" in data for _, data in suites)
    error_types = ("continuous", "grain_avg")
    if has_point_errors:
        error_types = error_types + ("point_errors",)

    ncols = len(error_types)
    fig, axes = plt.subplots(
        nrows=len(params),
        ncols=ncols,
        figsize=(5 * ncols, 8),
        constrained_layout=True,
    )
    fig.suptitle(_FIELD_TITLES[field], fontsize=14, fontweight="bold")

    ref_rate = 2.0 if field == "temperature" else 1.0

    for row, param in enumerate(params):
        # Rates row: shared y-limits across all columns.
        # Intercepts row: log scale, per-panel limits.
        shared_ymin, shared_ymax = (
            _param_limits(suites, field, param) if param == "rates" else (None, None)
        )

        for col, error_type in enumerate(error_types):
            ax = axes[row, col]

            # Skip point_errors scatter for suites that lack the data.
            active = [
                (name, data) for name, data in suites
                if error_type != "point_errors" or "point_errors" in data
            ]
            _scatter_ax(ax, active, error_type, field, param, shade=shade)

            ax.set_xlabel("grain volume (%)")
            ax.set_title(_ERROR_TITLES[error_type])
            ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

            if param == "intercepts":
                ax.set_yscale("log")
                ax.set_ylabel("coefficient  c  (log scale)")
                # Per-panel auto-ranging in log space; no explicit ylim.
            else:
                ax.set_ylabel(_PARAM_LABELS[param])
                ax.set_ylim(shared_ymin, shared_ymax)

            # Reference lines for the rates row only.
            if param == "rates":
                ax.axhline(0, color="red", linewidth=1.0, linestyle=":")
                ax.axhline(ref_rate, color="black", linewidth=1.0, linestyle="-")
                active_vals = np.concatenate([
                    data[error_type][field]["rates"] for _, data in active
                ])
                ax.axhline(active_vals.mean(), color="gray", linewidth=1.0,
                           linestyle=":")

    # Shared colorbar on the right.
    sm = plt.cm.ScalarMappable(cmap=_CMAP, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    fig.colorbar(sm, ax=axes, location="right", shrink=0.6,
                 label=_SHADE_LABEL[shade])

    return fig


def make_ridge_figure(suites, field, shade="R2"):
    """Build the 2 × N ridge-point figure for one field.

    One column per Voronoi suite that carries ``ridge_errors`` data; two
    rows for convergence rate (m) and coefficient (c).  Uses the same
    scatter/colour/reference-line style as :func:`make_figure`.

    The x-axis of each panel is the inter-seed distance of the Voronoi
    ridge, derived from the spacing of the 45 %/55 % evaluation points.

    Parameters
    ----------
    suites : list of (name, dict)
    field : "temperature" | "flux"
    shade : "R" | "R2"
        Quantity used for point colour.  Default ``"R2"``.

    Returns
    -------
    matplotlib.figure.Figure, or ``None`` if no suite has ridge data.
    """
    ridge_suites = [(name, data) for name, data in suites if "ridge_errors" in data]
    ncols = len(ridge_suites)
    if ncols == 0:
        return None

    fig, axes = plt.subplots(
        nrows=2,
        ncols=ncols,
        figsize=(4 * ncols, 8),
        constrained_layout=True,
    )
    # Guarantee a 2-D axes array even when ncols == 1.
    if ncols == 1:
        axes = axes[:, np.newaxis]

    fig.suptitle(
        f"{_FIELD_TITLES[field]} — ridge-point errors",
        fontsize=14, fontweight="bold",
    )

    params = ("rates", "intercepts")
    ref_rate = 2.0 if field == "temperature" else 1.0

    # Shared y-limits for the rates row, spanning all suites.
    all_rates = np.concatenate([
        d["ridge_errors"][field]["rates"] for _, d in ridge_suites
    ])
    shared_ylim = _limits(all_rates, "rates")

    for col, (suite_name, data) in enumerate(ridge_suites):
        rd = data["ridge_errors"]
        x  = rd["x"]

        for row, param in enumerate(params):
            ax     = axes[row, col]
            values = rd[field][param]
            r      = rd[field]["r"]

            ax.scatter(x, values, c=_shade_values(r, shade),
                       cmap=_CMAP, vmin=0, vmax=1, s=30, alpha=0.8)
            ax.set_xlabel("inter-seed distance")
            ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

            # Suite name as column header (top row only).
            if row == 0:
                ax.set_title(suite_name)

            # y-axis label on the leftmost column only.
            if col == 0:
                if param == "intercepts":
                    ax.set_ylabel("coefficient  c  (log scale)")
                else:
                    ax.set_ylabel(_PARAM_LABELS[param])

            if param == "intercepts":
                ax.set_yscale("log")
            else:
                ax.set_ylim(*shared_ylim)
                ax.axhline(0,             color="red",   linewidth=1.0, linestyle=":")
                ax.axhline(ref_rate,      color="black", linewidth=1.0, linestyle="-")
                ax.axhline(values.mean(), color="gray",  linewidth=1.0, linestyle=":")

    # Shared colorbar on the right.
    sm = plt.cm.ScalarMappable(cmap=_CMAP, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    fig.colorbar(sm, ax=axes, location="right", shrink=0.6,
                 label=_SHADE_LABEL[shade])

    return fig


# ------------------------------------------------------------------ #
# CLI                                                                  #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(
        description="Scatterplots of grain convergence parameters vs volume."
    )
    parser.add_argument("datafile", help="HDF5 file produced by run_convergence.py")
    parser.add_argument(
        "--show", action="store_true", help="display figures interactively"
    )
    parser.add_argument(
        "--ridge", action="store_true",
        help=(
            "produce additional figures showing ridge-point convergence fits "
            "(2 rows × N-suite cols, one figure per field)"
        ),
    )
    parser.add_argument(
        "--color-by", choices=["R", "R2"], default="R2",
        help="quantity used to shade scatter points: R or R² (default: R2)",
    )
    args = parser.parse_args()

    h5_path = Path(args.datafile)
    suites = load_all_suites(h5_path)

    stem = h5_path.stem
    out_dir = h5_path.parent
    shade = args.color_by

    for field in ("temperature", "flux"):
        fig = make_figure(suites, field, shade=shade)
        out_path = out_dir / f"{stem}-{field}.png"
        fig.savefig(out_path, dpi=150)
        print(f"saved → {out_path}")

    if args.ridge:
        for field in ("temperature", "flux"):
            fig = make_ridge_figure(suites, field, shade=shade)
            if fig is not None:
                out_path = out_dir / f"{stem}-ridge-{field}.png"
                fig.savefig(out_path, dpi=150)
                print(f"saved → {out_path}")
            else:
                print(f"no ridge_errors data found — skipping ridge figure for {field}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
