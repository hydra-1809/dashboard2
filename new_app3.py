import numpy as np
import pandas as pd
import streamlit as st

R = 8.314  # J/mol/K

st.set_page_config(page_title="Problem 13.18", page_icon="⚗️")
st.title("Problem 13.18 – Dehydrogenation of 1-Butene")
st.markdown(r"""
**C₄H₈(g) → CH₂=CH–CH=CH₂(g) + H₂(g)**  
Steam is added as an inert diluent. Equilibrium at T and P with a specified butadiene mole fraction in the product.
""")

# ---------------- Thermodynamic data (editable) ----------------
st.sidebar.header("Inputs")
T = st.sidebar.number_input("Temperature, T (K)", value=950.0, step=10.0)
P = st.sidebar.number_input("Pressure, P (bar)", value=1.0, step=0.1)
y_B = st.sidebar.number_input("Mole fraction of 1,3-butadiene in product", value=0.10,
                              min_value=0.001, max_value=0.45, step=0.01, format="%.3f")
P0 = 1.0  # standard state, bar
T0 = 298.15

st.sidebar.header("Thermo data (Smith, Van Ness & Abbott)")
data = pd.DataFrame(
    {
        "species": ["1-butene", "1,3-butadiene", "H2"],
        "nu": [-1, 1, 1],
        "dHf298": [-540.0, 109240.0, 0.0],      # J/mol
        "dGf298": [70240.0, 149790.0, 0.0],     # J/mol
        "A": [1.967, 2.734, 3.249],
        "B": [31.630e-3, 26.786e-3, 0.422e-3],
        "C": [-9.873e-6, -8.882e-6, 0.0],
        "D": [0.0, 0.0, 0.083e5],
    }
)
data = st.sidebar.data_editor(data, hide_index=True, use_container_width=True)

# ---------------- Reaction properties ----------------
nu = data["nu"].values
dH0 = float(np.sum(nu * data["dHf298"].values))
dG0 = float(np.sum(nu * data["dGf298"].values))
dA = float(np.sum(nu * data["A"].values))
dB = float(np.sum(nu * data["B"].values))
dC = float(np.sum(nu * data["C"].values))
dD = float(np.sum(nu * data["D"].values))


def K_at(T):
    """Equilibrium constant via Eq. 13.18 (SVNA), with integrals of dCp/R."""
    tau = T / T0
    # ∫ dCp/R dT  from T0 to T
    idcph = (dA * T0 * (tau - 1)
             + dB / 2 * T0**2 * (tau**2 - 1)
             + dC / 3 * T0**3 * (tau**3 - 1)
             + dD / T0 * (tau - 1) / tau)
    # ∫ dCp/R  dT/T  from T0 to T
    idcps = (dA * np.log(tau)
             + (dB * T0 + (dC * T0**2 + dD / (tau**2 * T0**2)) * (tau + 1) / 2) * (tau - 1))
    dG_RT = ((dG0 - dH0) / (R * T0) + dH0 / (R * T) + idcph / T - idcps)
    return np.exp(-dG_RT), dG_RT


K, dG_RT = K_at(T)

# ---------------- Solve ----------------
# K = (y_B * y_H / y_butene) * (P/P0)   (nu = +1),  y_H = y_B
y_butene = y_B**2 * (P / P0) / K
y_H = y_B
y_steam = 1.0 - y_B - y_H - y_butene

# Basis: 1 mol butene in feed
if y_steam < 0:
    st.error("No feasible solution: y_butene + y_B + y_H exceeds 1. "
             "Lower the butadiene mole fraction or change T/P.")
    st.stop()

N = 1.0 / (y_butene + y_B)          # total product moles
eps = y_B * N                       # extent of reaction
n_butene = 1 - eps
n_steam = N - 1 - eps               # steam moles (inert)
y_steam_feed = n_steam / (1 + n_steam)

# ---------------- Output ----------------
st.subheader("Reaction properties")
c1, c2, c3 = st.columns(3)
c1.metric("ΔH°₂₉₈ (J/mol)", f"{dH0:,.0f}")
c2.metric("ΔG°₂₉₈ (J/mol)", f"{dG0:,.0f}")
c3.metric(f"K at {T:.0f} K", f"{K:.4f}")

st.subheader("(a) Product gas composition")
comp = pd.DataFrame(
    {
        "Species": ["1-Butene", "1,3-Butadiene", "Hydrogen", "Steam"],
        "Mole fraction": [y_butene, y_B, y_H, y_steam],
        "Moles (per mol butene in feed)": [n_butene, eps, eps, n_steam],
    }
)
st.dataframe(comp.style.format({"Mole fraction": "{:.4f}",
                                "Moles (per mol butene in feed)": "{:.4f}"}),
             hide_index=True, use_container_width=True)

st.subheader("(b) Steam required in the feed")
c1, c2 = st.columns(2)
c1.metric("Feed mole fraction of steam", f"{y_steam_feed:.4f}")
c2.metric("Steam : butene (mol/mol)", f"{n_steam:.3f}")

with st.expander("Show working"):
    st.latex(r"K=\exp\!\left(-\frac{\Delta G^\circ}{RT}\right)=\frac{y_{B}\,y_{H_2}}{y_{butene}}\left(\frac{P}{P^\circ}\right)")
    st.write(f"ΔG°/RT at {T:.1f} K = {dG_RT:.4f}")
    st.write(f"ΔA = {dA:.3f}, ΔB = {dB:.4e}, ΔC = {dC:.4e}, ΔD = {dD:.1f}")
    st.write(f"y_butene = y_B² (P/P°) / K = {y_butene:.4f}")
    st.write(f"Basis 1 mol butene: N_total = {N:.4f}, ε = {eps:.4f}, steam = {n_steam:.4f} mol")
