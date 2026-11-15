import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import brentq

R = 8.314  # J/mol/K

st.set_page_config(page_title="Problem 13.31", page_icon="⚗️")
st.title("Problem 13.31 – Liquid-Phase Isomerization")
st.markdown(r"""
**A(l) → B(l)**, miscible liquids with $G^E/RT = a\,x_A x_B$.  
Find the equilibrium composition and the error from assuming an ideal solution.
""")

# ---------------- Inputs ----------------
st.sidebar.header("Inputs")
T = st.sidebar.number_input("Temperature, T (K)", value=298.15, step=5.0)
dG0 = st.sidebar.number_input("ΔG° (J/mol)", value=-1000.0, step=100.0)
a = st.sidebar.number_input("Margules parameter a  (G^E/RT = a·xA·xB)",
                            value=0.1, step=0.05, format="%.3f")

# ---------------- Calculations ----------------
K = np.exp(-dG0 / (R * T))


def ln_gammas(xB, a):
    xA = 1 - xB
    return a * xB**2, a * xA**2          # ln gamma_A, ln gamma_B


def residual(xB):
    lgA, lgB = ln_gammas(xB, a)
    return np.log(xB / (1 - xB)) + (lgB - lgA) - np.log(K)


xB_real = brentq(residual, 1e-10, 1 - 1e-10)
xA_real = 1 - xB_real
lgA, lgB = ln_gammas(xB_real, a)
gA, gB = np.exp(lgA), np.exp(lgB)

xB_ideal = K / (1 + K)
xA_ideal = 1 - xB_ideal

err_abs = xB_ideal - xB_real
err_pct = err_abs / xB_real * 100

# ---------------- Output ----------------
st.subheader("Equilibrium constant")
st.latex(r"K=\exp\!\left(-\frac{\Delta G^\circ}{RT}\right)")
st.metric(f"K at {T:.2f} K", f"{K:.4f}")

st.subheader("Equilibrium composition")
comp = pd.DataFrame(
    {
        "Model": ["Real (G^E/RT = a·xA·xB)", "Ideal solution"],
        "x_A": [xA_real, xA_ideal],
        "x_B": [xB_real, xB_ideal],
        "γ_A": [gA, 1.0],
        "γ_B": [gB, 1.0],
    }
)
st.dataframe(
    comp.style.format({"x_A": "{:.4f}", "x_B": "{:.4f}",
                       "γ_A": "{:.4f}", "γ_B": "{:.4f}"}),
    hide_index=True, use_container_width=True)

st.subheader("Error from assuming an ideal solution")
c1, c2, c3 = st.columns(3)
c1.metric("Δx_B (ideal − real)", f"{err_abs:+.4f}")
c2.metric("Error in x_B", f"{err_pct:+.2f} %")
c3.metric("Error in x_A", f"{(xA_ideal - xA_real) / xA_real * 100:+.2f} %")

# ---------------- Plot: Gibbs energy of mixture ----------------
st.subheader("Gibbs energy of the mixture vs composition")
st.caption("G/RT relative to pure A. The minimum is the equilibrium composition.")
x = np.linspace(0.001, 0.999, 500)
G_ideal = x * dG0 / (R * T) + (1 - x) * np.log(1 - x) + x * np.log(x)
G_real = G_ideal + a * x * (1 - x)
chart = pd.DataFrame({"x_B": x, "Ideal": G_ideal, "Real": G_real}).set_index("x_B")
st.line_chart(chart)

with st.expander("Show working"):
    st.latex(r"\ln\gamma_A=a\,x_B^2,\quad \ln\gamma_B=a\,x_A^2")
    st.latex(r"K=\frac{x_B\gamma_B}{x_A\gamma_A}"
             r"\;\Rightarrow\;\ln\frac{x_B}{1-x_B}+a(1-2x_B)=\ln K")
    st.write(f"ΔG°/RT = {dG0 / (R * T):.4f}")
    st.write(f"Ideal: x_B = K/(1+K) = {xB_ideal:.5f}")
    st.write(f"Real: x_B = {xB_real:.5f}, γ_A = {gA:.5f}, γ_B = {gB:.5f}")
