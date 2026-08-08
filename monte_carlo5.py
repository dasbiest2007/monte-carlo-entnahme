import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import locale
locale.setlocale(locale.LC_ALL, 'de_DE.UTF-8')

plt.rcParams['axes.formatter.use_locale'] = True
# ---------------------------------------------------------
# FARBCODES FÜR MONTE CARLO
# ---------------------------------------------------------
plt.style.use("dark_background")

COLOR_PATHS = "#3fa7ff"        # helle blaue Linien für Pfade
COLOR_MEDIAN_LINE = "#aaaaaa"  # graue Median-Linie
COLOR_P5 = "#ff5d73"           # rot für P5
COLOR_P95 = "#38d996"          # grün für P95

from matplotlib.colors import LinearSegmentedColormap

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
# Nominale Entnahme korrekt berechnen
# ---------------------------------------------------------

def nominal_entnahme(startkapital, zielwert, cagr, inflation, jahre):
    r_nom = cagr
    n = jahre

    # Summe der Barwerte aller Entnahmen
    faktor_summe = sum(
        (1 + r_nom)**(n - 1 - i) * (1 + inflation)**i
        for i in range(n)
    )

    # Anfangsentnahme nominal
    E0 = (startkapital * (1 + r_nom)**n - zielwert) / faktor_summe

    # Nominale Entnahmen über die Jahre
    entnahmen = np.array([E0 * (1 + inflation)**t for t in range(n)])

    # Kapitalverlauf nominal (vor Steuern)
    kapital = np.zeros(n + 1)
    kapital[0] = startkapital

    for t in range(1, n + 1):
        kapital[t] = kapital[t - 1] * (1 + r_nom) - entnahmen[t - 1]

    return E0, entnahmen, kapital

# ---------------------------------------------------------
# ENTNAHMEPLAN – real + nominal
# ---------------------------------------------------------

def entnahmeplan_nominal(startkapital, zielwert, cagr, inflation, jahre):
    r_nom = cagr
    n = jahre

    # Summe der Barwerte aller Entnahmen
    faktor_summe = sum(
        (1 + r_nom)**(n - 1 - i) * (1 + inflation)**i
        for i in range(n)
    )

    # Anfangsentnahme nominal
    E0 = (startkapital * (1 + r_nom)**n - zielwert) / faktor_summe

    jahre_liste = []
    entnahme_liste = []
    entnahme_monat_liste = []
    kapital_liste = []

    kapital = startkapital

    for t in range(n):
        entnahme_t = E0 * (1 + inflation)**t
        kapital = kapital * (1 + r_nom) - entnahme_t

        jahre_liste.append(t + 1)
        entnahme_liste.append(entnahme_t)
        entnahme_monat_liste.append(entnahme_t / 12)
        kapital_liste.append(kapital)

    return E0, jahre_liste, entnahme_liste, entnahme_monat_liste, kapital_liste

# ---------------------------------------------------------
# PLOT – ohne Steuern
# ---------------------------------------------------------

def plot_entnahmeplan_nominal(jahre_liste, kapital_liste):
    plt.style.use("dark_background")

    fig, ax = plt.subplots(figsize=(12, 6), dpi=110)

    ax.plot(jahre_liste, kapital_liste, label="Kapital (nominal)", linewidth=2.2, color="#ff7f24")

    ax.set_title("Nominaler Kapitalverlauf")
    ax.set_xlabel("Jahre")
    ax.set_ylabel("Kapital")
    ax.grid(True, alpha=0.3)
    ax.legend()

    return fig


# ---------------------------------------------------------
# BREAK-EVEN-INFLATION
# ---------------------------------------------------------

def break_even_inflation(startkapital, zielwert, cagr, jahre, low=0.0, high=0.2, tol=1e-4):
    def feasible(infl):
        er, en0, en, kr, kn = entnahmeplan(
            startkapital, zielwert, cagr, infl, jahre
        )
        return kr[-1] >= zielwert

    if feasible(high):
        return high

    for _ in range(60):
        mid = (low + high) / 2
        if feasible(mid):
            low = mid
        else:
            high = mid
        if high - low < tol:
            break

    return low

# ---------------------------------------------------------
# SZENARIEN – ohne Steuern
# ---------------------------------------------------------

def entnahme_szenarien_nominal(startkapital, zielwert, cagr, inflation, jahre):
    deltas = {
        "pessimistisch": cagr - 0.01,
        "realistisch": cagr,
        "optimistisch": cagr + 0.01
    }

    results = {}

    for name, c in deltas.items():
        E0, jahre_liste, entnahme_liste, entnahme_monat_liste, kapital_liste = entnahmeplan_nominal(
            startkapital, zielwert, c, inflation, jahre
        )

        results[name] = {
            "E0": E0,
            "jahre": jahre_liste,
            "entnahme_jahr": entnahme_liste,
            "entnahme_monat": entnahme_monat_liste,
            "kapital": kapital_liste
        }

    return results

# ---------------------------------------------------------
# MONTE CARLO SIMULATION – fehlender Teil
# ---------------------------------------------------------

def monte_carlo_simulation(
    initial_investment,
    target_cagr,
    volatility,
    years,
    monthly_contribution,
    num_simulations,
    steps_per_year=252,
    seed=None
):
    if seed is not None:
        np.random.seed(seed)

    total_steps = years * steps_per_year
    dt = 1 / steps_per_year

    # Lognormaler Drift
    drift = np.log(1 + target_cagr) * dt
    diffusion = volatility * np.sqrt(dt)

    # Zufallsbewegungen
    random_shocks = np.random.normal(0, 1, size=(num_simulations, total_steps))
    log_returns = drift + diffusion * random_shocks

    # Preis-Pfade
    price_paths = np.zeros((num_simulations, total_steps + 1))
    price_paths[:, 0] = initial_investment

    # Sparrate auf tägliche Schritte umrechnen
    contribution_per_step = monthly_contribution * 12 / steps_per_year

    for t in range(1, total_steps + 1):
        price_paths[:, t] = price_paths[:, t - 1] * np.exp(log_returns[:, t - 1]) + contribution_per_step

    final_values = price_paths[:, -1]

    # ---------------------------------------------------------
    # Stats – korrekt, sauber, funktioniert
    # ---------------------------------------------------------

    def fmt(x):
        return f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

    # Percentiles (numerisch)
    p5  = np.percentile(final_values, 5)
    p25 = np.percentile(final_values, 25)
    p50 = np.percentile(final_values, 50)
    p75 = np.percentile(final_values, 75)
    p95 = np.percentile(final_values, 95)

    # Wahrscheinlichkeiten (numerisch)
    # Gesamtinvestition berechnen
    total_contributions = monthly_contribution * 12 * years
    total_invested = initial_investment + total_contributions

    prob_loss    = np.mean(final_values < total_invested) * 100
    prob_double  = np.mean(final_values >= 2 * total_invested) * 100
    prob_tenfold = np.mean(final_values >= 10 * total_invested) * 100


    # Stats-Objekt – enthält BEIDES:
    # 1) numerische Werte für den Plot
    # 2) formatierte Strings für die Anzeige
    stats = {
        "percentiles": {
            5: p5,
            25: p25,
            50: p50,
            75: p75,
            95: p95
        },
        "formatted": {
            "P5": fmt(p5),
            "P25": fmt(p25),
            "Median": fmt(p50),
            "P75": fmt(p75),
            "P95": fmt(p95),
            "Wahrscheinlichkeit Verlust (%)": fmt(prob_loss),
            "Wahrscheinlichkeit Verdopplung (%)": fmt(prob_double),
            "Wahrscheinlichkeit Verzehnfachung (%)": fmt(prob_tenfold)
        }
    }

    return price_paths, final_values, stats


# ---------------------------------------------------------
# MONTE CARLO PLOT – vollständiger fehlender Teil
# ---------------------------------------------------------

def plot_simulation(price_paths, final_values, stats, years,
                    num_paths_to_plot=200, log_scale_paths=True,
                    hist_clip_percentile=99, hist_bins=80):

    fig, axes = plt.subplots(1, 2, figsize=(17, 6.5), dpi=110)

    # ---------- Chart 1 ----------
    time_axis = np.linspace(0, years, price_paths.shape[1])
    sample_indices = np.random.choice(
        price_paths.shape[0],
        min(num_paths_to_plot, price_paths.shape[0]),
        replace=False
    )

    ax0 = axes[0]
    for idx in sample_indices:
        ax0.plot(time_axis, price_paths[idx],
                 linewidth=0.5, alpha=0.10, color=COLOR_PATHS)

    p5 = np.percentile(price_paths, 5, axis=0)
    p25 = np.percentile(price_paths, 25, axis=0)
    p50 = np.percentile(price_paths, 50, axis=0)
    p75 = np.percentile(price_paths, 75, axis=0)
    p95 = np.percentile(price_paths, 95, axis=0)

    ax0.fill_between(time_axis, p5, p95, color="#ff5d73", alpha=0.12)
    ax0.fill_between(time_axis, p25, p75, color="#38d996", alpha=0.20)
    ax0.plot(time_axis, p50, color=COLOR_MEDIAN_LINE,
             linewidth=2.5, label="Median")

    ax0.set_title("Wertentwicklung (log)", pad=14)
    ax0.set_xlabel("Jahre")
    ax0.set_ylabel("Wert")
    ax0.xaxis.set_major_formatter(mticker.FuncFormatter(year_formatter))

    if log_scale_paths:
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
            bar_colors.append(GRADIENT_LEFT(np.sqrt(t)))

        elif p50_val < c <= p75_val:
            t = (c - p50_val) / (p75_val - p50_val)
            bar_colors.append(GRADIENT_RIGHT(np.sqrt(t)))

        else:
            bar_colors.append("#38d996")

    ax1.bar(bin_centers, counts,
            width=bin_edges[1] - bin_edges[0],
            color=bar_colors,
            edgecolor="#0f1117",
            linewidth=0.3)

    ax1.axvline(p50_val, color=COLOR_MEDIAN_LINE,
                linestyle="-", linewidth=1.8, label="Median")
    ax1.axvline(p5_val, color=COLOR_P5,
                linestyle=":", linewidth=2, label="P5")
    ax1.axvline(p95_val, color=COLOR_P95,
                linestyle=":", linewidth=2, label="P95")

    ax1.plot([], [], color="#38d996", linestyle="--", label="P25 (Farbgrenze)")
    ax1.plot([], [], color="#38d996", linestyle="--", label="P75 (Farbgrenze)")

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
