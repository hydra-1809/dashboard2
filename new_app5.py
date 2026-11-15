import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import brentq

R = 8.314       # J/mol/K
T0 = 298.15     # K
P0 = 1.0        # bar (standard state)
ATM_TO_BAR = 1.01325

st.set_page_config(page_title="Problem 13.14", page_icon="⚗️")
st.title("Problem 13.14 – Styrene Hydrogenation")
st.markdown(r"""
**C₆H₅CH=CH₂(g) + H₂(g) → C₆H₅C₂H₅(g)**  
Equilibrium composition for a given H₂ : styrene feed ratio. Ideal gases.
""")

# ---------------- Inputs ----------------
st.sidebar.header("Inputs")
T = st.sidebar.number_input("Temperature, T (K)", value=923.15, step=10.0)
P_atm = st.sidebar.number_input("Pressure (atm)", value=1.0, step=0.1)
ratio = st.sidebar.number_input("mol H₂ per mol styrene in feed", value=1.5,
                                min_value=0.01, step=0.1)
P = P_atm * ATM_TO_BAR

# ---------------- Data (editable) ----------------
st.sidebar.header("Thermo data (Smith, Van Ness & Abbott)")
data = pd.DataFrame(
    {
        "species": ["Styrene", "H2", "Ethylbenzene"],
        "nu": [-1, -1, 1],
        "dHf298": [147360.0, 0.0, 29920.0],
        "dGf298": [213900.0, 0.0, 130890.0],
        "A": [2.050, 3.249, 1.124],
        "B": [50.192e-3, 0.422e-3, 55.380e-3],
        "C": [-16.662e-6, 0.0, -18.476e-6],
        "D": [0.0, 0.083e5, 0.0],
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


def solve_extent(T, P, ratio):
    """Extent of reaction per mol styrene fed."""
    K, _ = K_at(T)
    n_S0, n_H0 = 1.0, ratio
    n_tot0 = n_S0 + n_H0

    def f(e):
        # K*(P/P0) = e*(n_tot0 - e) / ((n_S0 - e)(n_H0 - e))
        return e * (n_tot0 - e) / ((n_S0 - e) * (n_H0 - e)) - K * P / P0

    e_max = min(n_S0, n_H0)
    return brentq(f, 1e-12, e_max * (1 - 1e-12))


K, dG_RT = K_at(T)
eps = solve_extent(T, P, ratio)

n_S = 1.0 - eps
n_H = ratio - eps
n_E = eps
n_tot = 1.0 + ratio - eps

# ---------------- Output ----------------
st.subheader("Reaction properties")
c1, c2, c3 = st.columns(3)
c1.metric("ΔH°₂₉₈ (J/mol)", f"{dH0:,.0f}")
c2.metric("ΔG°₂₉₈ (J/mol)", f"{dG0:,.0f}")
c3.metric(f"K at {T:.2f} K", f"{K:.4f}")

st.subheader("Equilibrium composition")
c1, c2 = st.columns(2)
c1.metric("Extent of reaction, ε", f"{eps:.4f}")
c2.metric("Styrene conversion", f"{eps * 100:.2f} %")

comp = pd.DataFrame(
    {
        "Species": ["Styrene", "Hydrogen", "Ethylbenzene"],
        "Moles (per mol styrene fed)": [n_S, n_H, n_E],
        "Mole fraction": [n_S / n_tot, n_H / n_tot, n_E / n_tot],
    }
)
st.dataframe(
    comp.style.format({"Moles (per mol styrene fed)": "{:.4f}",
                       "Mole fraction": "{:.4f}"}),
    hide_index=True, use_container_width=True)

# ---------------- Plots ----------------
st.subheader("Conversion vs temperature")
Ts = np.linspace(600, 1200, 100)
conv = [solve_extent(t, P, ratio) for t in Ts]
st.line_chart(pd.DataFrame({"T (K)": Ts, "Styrene conversion": conv}).set_index("T (K)"))

with st.expander("Show working"):
    st.latex(r"K=\frac{y_{EB}}{y_{S}\,y_{H_2}}\frac{P^\circ}{P}"
             r"\;\Rightarrow\;K\frac{P}{P^\circ}=\frac{\varepsilon\,(2.5-\varepsilon)}{(1-\varepsilon)(1.5-\varepsilon)}")
    st.write(f"ΔA = {dA:.3f}, ΔB = {dB:.4e}, ΔC = {dC:.4e}, ΔD = {dD:.1f}")
    st.write(f"ΔG°/RT at {T:.2f} K = {dG_RT:.4f}")
    st.write(f"P = {P:.5f} bar")
    st.write(f"Total moles at equilibrium = {n_tot:.4f}")
