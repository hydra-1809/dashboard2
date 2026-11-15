import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import brentq

R = 8.314       # J/mol/K
T0 = 298.15     # K
P0 = 1.0        # bar

st.set_page_config(page_title="Problem 13.29", page_icon="⚗️")
st.title("Problem 13.29 – H₂S + SO₂ Claus Reaction")
st.markdown(r"""
**2H₂S(g) + SO₂(g) → 3S(s) + 2H₂O(g)**  
Stoichiometric feed (2 H₂S : 1 SO₂). Solid sulfur is a pure phase (activity = 1).
""")

# ---------------- Inputs ----------------
st.sidebar.header("Inputs")
T = st.sidebar.number_input("Temperature, T (K)", value=723.15, step=10.0)
P = st.sidebar.number_input("Pressure, P (bar)", value=8.0, step=0.5)

# ---------------- Data (editable) ----------------
st.sidebar.header("Thermo data (Smith, Van Ness & Abbott)")
data = pd.DataFrame(
    {
        "species": ["H2S", "SO2", "S(s)", "H2O(g)"],
        "nu": [-2, -1, 3, 2],
        "gas": [True, True, False, True],
        "dHf298": [-20630.0, -296830.0, 0.0, -241818.0],
        "dGf298": [-33560.0, -300194.0, 0.0, -228572.0],
        "A": [3.931, 5.699, 2.72, 3.470],
        "B": [1.490e-3, 0.801e-3, 0.0, 1.450e-3],
        "C": [0.0, 0.0, 0.0, 0.0],
        "D": [-0.232e5, -1.015e5, 0.0, 0.121e5],
    }
)
data = st.sidebar.data_editor(data, hide_index=True, use_container_width=True)

nu = data["nu"].values
dH0 = float(np.sum(nu * data["dHf298"].values))
dG0 = float(np.sum(nu * data["dGf298"].values))
dA = float(np.sum(nu * data["A"].values))
dB = float(np.sum(nu * data["B"].values))
dC = float(np.sum(nu * data["C"].values))
dD = float(np.sum(nu * data["D"].values))
dnu_gas = float(np.sum(nu[data["gas"].values.astype(bool)]))   # = -1


def K_at(T):
    """Equilibrium constant, SVNA Eq. 13.18 with integrals of dCp/R."""
    tau = T / T0
    idcph = (dA * T0 * (tau - 1)
             + dB / 2 * T0**2 * (tau**2 - 1)
             + dC / 3 * T0**3 * (tau**3 - 1)
             + dD / T0 * (tau - 1) / tau)
    idcps = (dA * np.log(tau)
             + (dB * T0 + (dC * T0**2 + dD / (tau**2 * T0**2)) * (tau + 1) / 2) * (tau - 1))
    dG_RT = (dG0 - dH0) / (R * T0) + dH0 / (R * T) + idcph / T - idcps
    return float(np.exp(-dG_RT)), float(dG_RT)


def solve_extent(T, P):
    """Conversion of each reactant (stoichiometric feed, basis 2 H2S + 1 SO2)."""
    K, _ = K_at(T)
    rhs = K * (P / P0) ** (-dnu_gas)      # dnu_gas = -1 -> K*P/P0... see below

    # K = [e^2 (3-e) / (1-e)^3] * (P0/P)  ->  e^2(3-e)/(1-e)^3 = K * P/P0
    f = lambda e: e**2 * (3 - e) / (1 - e) ** 3 - K * P / P0
    return brentq(f, 1e-12, 1 - 1e-14)


K, dG_RT = K_at(T)
eps = solve_extent(T, P)

n_H2S = 2 - 2 * eps
n_SO2 = 1 - eps
n_H2O = 2 * eps
n_S = 3 * eps
n_gas = 3 - eps

# ---------------- Output ----------------
st.subheader("Reaction properties")
c1, c2, c3 = st.columns(3)
c1.metric("ΔH°₂₉₈ (J/mol)", f"{dH0:,.0f}")
c2.metric("ΔG°₂₉₈ (J/mol)", f"{dG0:,.0f}")
c3.metric(f"K at {T:.2f} K", f"{K:.4e}")

st.subheader("Percent conversion at equilibrium")
c1, c2 = st.columns(2)
c1.metric("H₂S conversion", f"{eps * 100:.3f} %")
c2.metric("SO₂ conversion", f"{eps * 100:.3f} %")

comp = pd.DataFrame(
    {
        "Species": ["H2S", "SO2", "H2O", "S(s)"],
        "Moles (basis 2 H2S + 1 SO2)": [n_H2S, n_SO2, n_H2O, n_S],
        "Gas mole fraction": [n_H2S / n_gas, n_SO2 / n_gas, n_H2O / n_gas, np.nan],
    }
)
st.dataframe(
    comp.style.format({"Moles (basis 2 H2S + 1 SO2)": "{:.5f}",
                       "Gas mole fraction": "{:.5f}"}, na_rep="-"),
    hide_index=True, use_container_width=True)

# ---------------- Plot ----------------
st.subheader("Conversion vs temperature")
Ts = np.linspace(400, 900, 101)
conv = [solve_extent(t, P) * 100 for t in Ts]
st.line_chart(pd.DataFrame({"T (K)": Ts, "Conversion (%)": conv}).set_index("T (K)"))

with st.expander("Show working"):
    st.latex(r"K=\frac{y_{H_2O}^2}{y_{H_2S}^2\,y_{SO_2}}\frac{P^\circ}{P}"
             r"\;\Rightarrow\;\frac{\varepsilon^2(3-\varepsilon)}{(1-\varepsilon)^3}=K\frac{P}{P^\circ}")
    st.write(f"ΔA = {dA:.3f}, ΔB = {dB:.4e}, ΔC = {dC:.4e}, ΔD = {dD:.1f}")
    st.write(f"ΔG°/RT at {T:.2f} K = {dG_RT:.4f}")
    st.write(f"K·P/P° = {K * P / P0:.4e}")
