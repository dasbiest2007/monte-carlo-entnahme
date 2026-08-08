import streamlit as st
from monte_carlo5 import (
    monte_carlo_simulation,
    plot_simulation,
    entnahmeplan_nominal,
    plot_entnahmeplan_nominal,
    entnahme_szenarien_nominal,
    break_even_inflation
)

def fmt(x):
    return f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


st.set_page_config(page_title="Finanz‑Tool", layout="wide")

st.title("Finanz‑Tool")

option = st.selectbox(
    "Was möchtest du berechnen?",
    ["Monte‑Carlo Simulation", "Entnahmeplan"]
)

# ---------------------------------------------------------
# MONTE CARLO
# ---------------------------------------------------------

# ---------------------------------------------------------
# MONTE CARLO
# ---------------------------------------------------------

# ---------------------------------------------------------
# MONTE CARLO
# ---------------------------------------------------------

if option == "Monte‑Carlo Simulation":
    st.header("Monte‑Carlo Simulation")

    initial_investment = st.number_input("Anfangsinvestition (€)", value=10_000)
    target_cagr = st.number_input("Erwartete Rendite (CAGR)", value=0.08)
    volatility = st.number_input("Volatilität p.a.", value=0.15)
    years = st.number_input("Anlagehorizont (Jahre)", value=25)
    monthly_contribution = st.number_input("Monatliche Sparrate (€)", value=300)
    num_simulations = st.number_input("Anzahl der Simulationen", value=15_000)
    steps_per_year = st.number_input("Zeitschritte pro Jahr", value=252)
    seed = st.number_input("Seed (0 = Zufall)", value=0)

    if st.button("Simulation starten"):

        # ---- Monte-Carlo Simulation ----
        price_paths, final_values, stats = monte_carlo_simulation(
            initial_investment,
            target_cagr,
            volatility,
            years,
            monthly_contribution,
            num_simulations
        )

        # ---- Grafik zuerst anzeigen ----
        fig = plot_simulation(price_paths, final_values, stats, years)
        st.pyplot(fig)

        # ---- Danach Stats anzeigen ----
        f = stats["formatted"]

        st.subheader("Kennzahlen")

        st.write(f"P5: {f['P5']} €")
        st.write(f"P25: {f['P25']} €")
        st.write(f"Median: {f['Median']} €")
        st.write(f"P75: {f['P75']} €")
        st.write(f"P95: {f['P95']} €")

        st.write("---")

        st.write(f"Wahrscheinlichkeit Verlust: {f['Wahrscheinlichkeit Verlust (%)']} %")
        st.write(f"Wahrscheinlichkeit Verdopplung: {f['Wahrscheinlichkeit Verdopplung (%)']} %")
        st.write(f"Wahrscheinlichkeit Verzehnfachung: {f['Wahrscheinlichkeit Verzehnfachung (%)']} %")



# ---------------------------------------------------------
# ENTNAHMEPLAN
# ---------------------------------------------------------

if option == "Entnahmeplan":
    st.header("Entnahmeplan (nur nominal)")

    startkapital = st.number_input("Startkapital (€)", value=300_000)
    zielwert = st.number_input("Zielendwert (€)", value=0)
    cagr = st.number_input("Erwartete Rendite (CAGR)", value=0.035)
    inflation = st.number_input("Jährliche Inflation", value=0.02)
    jahre = st.number_input("Entnahmezeitraum (Jahre)", value=30)

    if st.button("Entnahme berechnen"):
        (
        E0,
        jahre_liste,
        entnahme_jahr,
        entnahme_monat,
        kapital_liste
    ) = entnahmeplan_nominal(startkapital, zielwert, cagr, inflation, jahre)

    import pandas as pd

    df = pd.DataFrame({
        "Jahr": jahre_liste,
        "Entnahme nominal (Jahr)": [fmt(x) for x in entnahme_jahr],
        "Entnahme nominal (Monat)": [fmt(x) for x in entnahme_monat],
        "Kapital am Jahresende": [fmt(x) for x in kapital_liste]
    })

    st.dataframe(df.set_index("Jahr"))

    fig = plot_entnahmeplan_nominal(jahre_liste, kapital_liste)
    st.pyplot(fig)


    

    