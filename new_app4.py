import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import brentq

R = 8.314      # J/mol/K
T0 = 298.15    # K
P0 = 1.0       # bar

st.set_page_config(page_title="Problem 13.20", page_icon="⚗️")
st.title("Problem 13.20 – Ammonia Synthesis")
st.markdown(r"""
**½N₂(g) + 3/2 H₂(g) → NH₃(g)**, feed in stoichiometric proportion (H₂:N₂ = 3:1).
""")

# ---------------- Data (editable) ----------------
st.sidebar.header("Thermo / critical data")
data = pd.DataFrame(
    {
        "species": ["N2", "H2", "NH3"],
        "nu": [-0.5, -1.5, 1.0],
        "dHf298": [0.0, 0.0, -46110.0],
        "dGf298": [0.0, 0.0, -16450.0],
        "A": [3.280, 3.249, 3.578],
        "B": [0.593e-3, 0.422e-3, 3.020e-3],
        "C": [0.0, 0.0, 0.0],
        "D": [0.040e5, 0.083e5, -0.186e5],
        "Tc": [126.2, 33.19, 405.7],
        "Pc": [34.00, 13.13, 112.8],
        "omega": [0.038, -0.216, 0.253],
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
    return float(np.exp(-dG_RT))


def phi_pure(T, P):
    """Pure-species fugacity coefficients, Pitzer virial correlation."""
    Tr = T / data["Tc"].values
    Pr = P / data["Pc"].values
    w = data["omega"].values
    B0 = 0.083 - 0.422 / Tr**1.6
    B1 = 0.139 - 0.172 / Tr**4.2
    return np.exp(Pr / Tr * (B0 + w * B1))


def phi_ratio(T, P):
    """prod(phi_i^nu_i) for the ideal-solution model."""
    return float(np.prod(phi_pure(T, P) ** nu))


def y_NH3(T, P, ideal_solution=False):
    """Equilibrium NH3 mole fraction for stoichiometric feed."""
    K = K_at(T)
    if ideal_solution:
        K = K / phi_ratio(T, P)        # K_y-type constant: K / prod(phi^nu)
    rhs = K * P / P0 * 0.5 * 0.75**1.5 * 1.0   # y/(1-y)^2 = rhs
    # y / ((0.25(1-y))^0.5 (0.75(1-y))^1.5) = K*P/P0  ->  y/(1-y)^2 = K*P/P0 * 0.5*0.75^1.5
    f = lambda y: y / (1 - y) ** 2 - rhs
    return brentq(f, 1e-12, 1 - 1e-12)


def T_for_y(y_target, P, ideal_solution=False, Tlo=250.0, Thi=1500.0):
    f = lambda T: y_NH3(T, P, ideal_solution) - y_target
    return brentq(f, Tlo, Thi)


# ---------------- Inputs ----------------
st.sidebar.header("Inputs")
T_a = st.sidebar.number_input("(a) Temperature (K)", value=300.0, step=10.0)
P_a = st.sidebar.number_input("(a) Pressure (bar)", value=1.0, step=1.0)
y_t = st.sidebar.number_input("Target NH₃ mole fraction (b–d)", value=0.50,
                              min_value=0.01, max_value=0.99, step=0.05)
P_b = st.sidebar.number_input("(b) Pressure (bar)", value=1.0, step=1.0)
P_cd = st.sidebar.number_input("(c, d) Pressure (bar)", value=100.0, step=10.0)

# ---------------- Reaction properties ----------------
st.subheader("Reaction properties")
c1, c2, c3 = st.columns(3)
c1.metric("ΔH°₂₉₈ (J/mol)", f"{dH0:,.0f}")
c2.metric("ΔG°₂₉₈ (J/mol)", f"{dG0:,.0f}")
c3.metric("K₂₉₈", f"{K_at(298.15):.2f}")

# ---------------- (a) ----------------
st.subheader(f"(a) Equilibrium y(NH₃) at {P_a:g} bar and {T_a:g} K")
ya = y_NH3(T_a, P_a)
c1, c2 = st.columns(2)
c1.metric(f"K at {T_a:g} K", f"{K_at(T_a):.3f}")
c2.metric("y(NH₃)", f"{ya:.4f}")

# ---------------- (b) ----------------
st.subheader(f"(b) T for y(NH₃) = {y_t:g} at {P_b:g} bar")
Tb = T_for_y(y_t, P_b)
st.metric("Temperature", f"{Tb:.1f} K  ({Tb - 273.15:.1f} °C)")

# ---------------- (c) ----------------
st.subheader(f"(c) T for y(NH₃) = {y_t:g} at {P_cd:g} bar, ideal gas")
Tc_ = T_for_y(y_t, P_cd)
st.metric("Temperature", f"{Tc_:.1f} K  ({Tc_ - 273.15:.1f} °C)")

# ---------------- (d) ----------------
st.subheader(f"(d) T for y(NH₃) = {y_t:g} at {P_cd:g} bar, ideal solution of real gases")
Td = T_for_y(y_t, P_cd, ideal_solution=True)
phis = phi_pure(Td, P_cd)
st.metric("Temperature", f"{Td:.1f} K  ({Td - 273.15:.1f} °C)")
st.dataframe(
    pd.DataFrame({"Species": data["species"],
                  "φ (pure, T, P)": phis}).style.format({"φ (pure, T, P)": "{:.4f}"}),
    hide_index=True, use_container_width=True)

# ---------------- Plot ----------------
st.subheader("y(NH₃) vs T")
Ts = np.linspace(300, 900, 121)
chart = pd.DataFrame({
    "T (K)": Ts,
    f"{P_b:g} bar, ideal gas": [y_NH3(t, P_b) for t in Ts],
    f"{P_cd:g} bar, ideal gas": [y_NH3(t, P_cd) for t in Ts],
    f"{P_cd:g} bar, ideal solution": [y_NH3(t, P_cd, True) for t in Ts],
}).set_index("T (K)")
st.line_chart(chart)

with st.expander("Show working"):
    st.latex(r"K=\frac{y_{NH_3}}{y_{N_2}^{1/2}y_{H_2}^{3/2}}\frac{P^\circ}{P}"
             r"\;\;\Rightarrow\;\;\frac{y}{(1-y)^2}=0.3248\,K\,\frac{P}{P^\circ}")
    st.latex(r"\text{Ideal solution: } K=\frac{y_{NH_3}}{y_{N_2}^{1/2}y_{H_2}^{3/2}}"
             r"\frac{\phi_{NH_3}}{\phi_{N_2}^{1/2}\phi_{H_2}^{3/2}}\frac{P^\circ}{P}")
    st.write(f"ΔA = {dA:.3f}, ΔB = {dB:.4e}, ΔC = {dC:.4e}, ΔD = {dD:.1f}")
    st.write(f"K at {Td:.1f} K = {K_at(Td):.4f};  φ-ratio = {phi_ratio(Td, P_cd):.4f}")
