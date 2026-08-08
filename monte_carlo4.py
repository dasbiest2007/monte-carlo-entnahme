import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap

# ---------------------------------------------------------
# STYLE
# ---------------------------------------------------------

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titleweight": "bold",
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "axes.edgecolor": "#444444",
    "axes.linewidth": 1.1,
    "figure.facecolor": "#0f1117",
    "axes.facecolor": "#0f1117",
    "savefig.facecolor": "#0f1117",
    "text.color": "#e8e8e8",
    "axes.labelcolor": "#e8e8e8",
    "xtick.color": "#c5c5c5",
    "ytick.color": "#c5c5c5",
    "axes.titlecolor": "#ffffff",
    "grid.color": "#333844",
})

COLOR_PATHS = "#3fa7ff"
COLOR_MEDIAN_LINE = "#aaaaaa"
COLOR_P5 = "#ff5d73"
COLOR_P95 = "#38d996"

GRADIENT_LEFT = LinearSegmentedColormap.from_list(
    "grad_left",
    [
        (0.00, "#38d996"),
        (0.20, "#d4f442"),
        (0.40, "#f4e842"),
        (0.60, "#ffb347"),
        (0.80, "#ff7f24"),
        (1.00, "#ff3b30"),
    ]
)

GRADIENT_RIGHT = LinearSegmentedColormap.from_list(
    "grad_right",
    [
        (0.00, "#ff3b30"),
        (0.20, "#ff7f24"),
        (0.40, "#ffb347"),
        (0.60, "#f4e842"),
        (0.80, "#d4f442"),
        (1.00, "#38d996"),
    ]
)

# ---------------------------------------------------------
# FORMATTER
# ---------------------------------------------------------

def euro_formatter(x, pos=None):
    if abs(x) >= 1_000_000:
        return f"{x/1_000_000:,.1f} Mio. €".replace(",", "X").replace(".", ",").replace("X", ".")
    if abs(x) >= 1_000:
        return f"{x/1_000:,.0f} Tsd. €".replace(",", ".")
    return f"{x:,.0f} €".replace(",", ".")

def year_formatter(x, pos=None):
    return f"{x:.0f} J."

# ---------------------------------------------------------
# MONTE CARLO SIMULATION
# ---------------------------------------------------------

def monte_carlo_simulation(
    initial_investment,
    target_cagr,
    volatility,
    years,
    monthly_contribution,
    num_simulations,
    steps_per_year=252,
    confidence_intervals=(5, 25, 50, 75, 95),
    seed=None
):
    if seed is not None:
        np.random.seed(seed)

    total_steps = years * steps_per_year
    dt = 1 / steps_per_year
    sigma = volatility

    drift = np.log(1 + target_cagr) * dt
    diffusion = sigma * np.sqrt(dt)

    random_shocks = np.random.normal(0, 1, size=(num_simulations, total_steps))
    log_returns = drift + diffusion * random_shocks

    price_paths = np.zeros((num_simulations, total_steps + 1))
    price_paths[:, 0] = initial_investment

    contribution_per_step = monthly_contribution * 12 / steps_per_year

    for t in range(1, total_steps + 1):
        price_paths[:, t] = price_paths[:, t - 1] * np.exp(log_returns[:, t - 1]) + contribution_per_step

    total_contributions = initial_investment + monthly_contribution * 12 * years
    final_values = price_paths[:, -1]

    stats = {
        "mean_final_value": np.mean(final_values),
        "median_final_value": np.median(final_values),
        "std_final_value": np.std(final_values),
        "min_final_value": np.min(final_values),
        "max_final_value": np.max(final_values),
        "total_contributions": total_contributions,
        "percentiles": {p: np.percentile(final_values, p) for p in confidence_intervals},
        "prob_loss": np.mean(final_values < total_contributions) * 100,
        "prob_double": np.mean(final_values >= 2 * total_contributions) * 100,
        "target_cagr": target_cagr,
        "realized_median_cagr": (np.median(final_values) / initial_investment) ** (1 / years) - 1,
        "realized_mean_cagr": (np.mean(final_values) / initial_investment) ** (1 / years) - 1,
        "var_95": total_contributions - np.percentile(final_values, 5),
        "cvar_95": total_contributions - np.mean(final_values[final_values <= np.percentile(final_values, 5)]),
    }

    return price_paths, final_values, stats

# ---------------------------------------------------------
# ENTNAHMEPLAN MIT INFLATION
# ---------------------------------------------------------

def entnahmeplan_rechner(startkapital, zielwert, cagr, inflation, jahre):
    monate = jahre * 12

    r_monat = (1 + cagr) ** (1/12) - 1
    i_monat = (1 + inflation) ** (1/12) - 1

    r_real = (1 + r_monat) / (1 + i_monat) - 1

    faktor_summe = sum((1 + r_real) ** (monate - i) for i in range(monate))

    entnahme = (startkapital * (1 + r_real) ** monate - zielwert) / faktor_summe

    kapital = np.zeros(monate + 1)
    kapital[0] = startkapital

    for m in range(1, monate + 1):
        kapital[m] = kapital[m-1] * (1 + r_real) - entnahme

    fig, ax = plt.subplots(figsize=(12, 6), dpi=110)

    ax.plot(np.arange(monate+1)/12, kapital, color=COLOR_PATHS, linewidth=2.2)
    ax.axhline(zielwert, color=COLOR_MEDIAN_LINE, linestyle="--", linewidth=1.8)

    ax.set_title("Entnahmeplan – Kapitalverlauf (inflationsbereinigt)")
    ax.set_xlabel("Jahre")
    ax.set_ylabel("Kapital")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(year_formatter))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(euro_formatter))
    ax.grid(True, alpha=0.5)

    plt.tight_layout()
    return fig, entnahme, kapital[-1]

# ---------------------------------------------------------
# MONTE CARLO PLOT
# ---------------------------------------------------------

def plot_simulation(price_paths, final_values, stats, years,
                    num_paths_to_plot=200, log_scale_paths=True,
                    hist_clip_percentile=99, hist_bins=80):

    fig, axes = plt.subplots(1, 2, figsize=(17, 6.5), dpi=110)

    # ---------- Chart 1 ----------
    time_axis = np.linspace(0, years, price_paths.shape[1])
    sample_indices = np.random.choice(price_paths.shape[0], min(num_paths_to_plot, price_paths.shape[0]), replace=False)

    ax0 = axes[0]
    for idx in sample_indices:
        ax0.plot(time_axis, price_paths[idx], linewidth=0.5, alpha=0.10, color=COLOR_PATHS)

    p5 = np.percentile(price_paths, 5, axis=0)
    p25 = np.percentile(price_paths, 25, axis=0)
    p50 = np.percentile(price_paths, 50, axis=0)
    p75 = np.percentile(price_paths, 75, axis=0)
    p95 = np.percentile(price_paths, 95, axis=0)

    ax0.fill_between(time_axis, p5, p95, color="#ff5d73", alpha=0.12)
    ax0.fill_between(time_axis, p25, p75, color="#38d996", alpha=0.20)
    ax0.plot(time_axis, p50, color=COLOR_MEDIAN_LINE, linewidth=2.5, label="Median")

    ax0.set_title("Wertentwicklung (log)", pad=14)
    ax0.set_xlabel("Jahre")
    ax0.set_ylabel("Wert")
    ax0.xaxis.set_major_formatter(mticker.FuncFormatter(year_formatter))
    ax0.set_yscale("log")
    ax0.yaxis.set_major_formatter(mticker.FuncFormatter(euro_formatter))
    ax0.grid(True, alpha=0.25, linestyle="--", linewidth=0.7)
    ax0.legend(frameon=False)

    # ---------- Chart 2 ----------
    ax1 = axes[1]

    x_max = np.percentile(final_values, hist_clip_percentile)
    plotted_values = final_values[final_values <= x_max]

    counts, bin_edges = np.histogram(plotted_values, bins=hist_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    p5_val = stats["percentiles"][5]
    p25_val = stats["percentiles"][25]
    p50_val = stats["percentiles"][50]
    p75_val = stats["percentiles"][75]
    p95_val = stats["percentiles"][95]

    median_bin_idx = np.argmin(np.abs(bin_centers - p50_val))

    bar_colors = []
    for i, c in enumerate(bin_centers):

        if i == median_bin_idx:
            bar_colors.append("#ff3b30")

        elif c < p25_val or c > p75_val:
            bar_colors.append("#38d996")

        elif p25_val <= c < p50_val:
            t = (c - p25_val) / (p50_val - p25_val)
            t_strong = np.sqrt(t)
            bar_colors.append(GRADIENT_LEFT(t_strong))

        elif p50_val < c <= p75_val:
            t = (c - p50_val) / (p75_val - p50_val)
            t_strong = np.sqrt(t)
            bar_colors.append(GRADIENT_RIGHT(t_strong))

        else:
            bar_colors.append("#38d996")

    ax1.bar(bin_centers, counts, width=bin_edges[1] - bin_edges[0],
            color=bar_colors, edgecolor="#0f1117", linewidth=0.3)

    ax1.axvline(p50_val, color=COLOR_MEDIAN_LINE, linestyle="-", linewidth=1.8, label="Median")
    ax1.axvline(p5_val, color=COLOR_P5, linestyle=":", linewidth=2, label="P5")
    ax1.axvline(p95_val, color=COLOR_P95, linestyle=":", linewidth=2, label="P95")

    ax1.plot([], [], color="#38d996", linestyle="--", label="P25 (Farbgrenze)")
    ax1.plot([], [], color="#38d996", linestyle="--", label="P75 (Farbgrenze)")

    # ⭐ Textblock rechts oben
    median_de = f"{p50_val:,.0f}".replace(",", "X").replace(".", ",").replace("X", ".")
    p25_de   = f"{p25_val:,.0f}".replace(",", "X").replace(".", ",").replace("X", ".")
    p75_de   = f"{p75_val:,.0f}".replace(",", "X").replace(".", ",").replace("X", ".")

    ax1.text(
        0.95, 0.60,
        f"Median: {median_de} €\n"
        f"P25:    {p25_de} €\n"
        f"P75:    {p75_de} €",
        transform=ax1.transAxes,
        fontsize=11,
        color=COLOR_MEDIAN_LINE,
        ha="right",
        va="top"
    )

    ax1.set_title("Endwerte", pad=14)
    ax1.set_xlabel("Wert")
    ax1.set_ylabel("Häufigkeit")
    ax1.xaxis.set_major_formatter(mticker.FuncFormatter(euro_formatter))
    ax1.grid(True, axis="y", alpha=0.25, linestyle="--", linewidth=0.7)
    ax1.legend(frameon=False, loc="upper right")

    plt.tight_layout()
    return fig
