import streamlit as st
from monte_carlo4 import (
    monte_carlo_simulation,
    plot_simulation,
    entnahmeplan_rechner
)

st.set_page_config(page_title="Finanz‑Tool", layout="wide")

st.title("Finanz‑Tool")

option = st.selectbox(
    "Was möchtest du berechnen?",
    ["Monte‑Carlo Simulation", "Entnahmeplan"]
)

# ---------------------------------------------------------
# MONTE CARLO
# ---------------------------------------------------------

if option == "Monte‑Carlo Simulation":
    st.header("Monte‑Carlo Simulation")

    initial_investment = st.number_input("Anfangsinvestition (€)", value=10000)
    target_cagr = st.number_input("Erwartete Rendite (CAGR)", value=0.08)
    volatility = st.number_input("Volatilität p.a.", value=0.15)
    years = st.number_input("Anlagehorizont (Jahre)", value=25)
    monthly_contribution = st.number_input("Monatliche Sparrate (€)", value=300)
    num_simulations = st.number_input("Anzahl der Simulationen", value=5000)

    if st.button("Simulation starten"):
        price_paths, final_values, stats = monte_carlo_simulation(
            initial_investment,
            target_cagr,
            volatility,
            years,
            monthly_contribution,
            num_simulations
        )

        fig = plot_simulation(price_paths, final_values, stats, years)
        st.pyplot(fig)

        st.subheader("Ergebnisse")
        st.write(stats)

# ---------------------------------------------------------
# ENTNAHMEPLAN
# ---------------------------------------------------------

if option == "Entnahmeplan":
    st.header("Entnahmeplan")

    startkapital = st.number_input("Startkapital (€)", value=300000)
    zielwert = st.number_input("Zielendwert (€)", value=0)
    cagr = st.number_input("Erwartete Rendite (CAGR)", value=0.035)
    jahre = st.number_input("Entnahmezeitraum (Jahre)", value=30)

    if st.button("Entnahme berechnen"):
        fig, entnahme, endkapital = entnahmeplan_rechner(startkapital, zielwert, cagr, jahre)

        st.pyplot(fig)

        st.subheader("Ergebnisse")
        st.write(f"Entnahme pro Monat: {entnahme:,.2f} €")
        st.write(f"Entnahme pro Jahr: {entnahme*12:,.2f} €")

