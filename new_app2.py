import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import math

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Q12.13 | VLE Dashboard",
    page_icon="⚗️",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.stApp {
    background: #0b1220;
    color: #e5e7eb;
}
.block-container {
    padding-top: 2rem;
    max-width: 1400px;
}
h1, h2, h3 {
    color: #e2e8f0 !important;
}
div[data-testid="stMetric"] {
    background: #172338;
    padding: 16px;
    border-radius: 12px;
    border: 1px solid #293b56;
}
section[data-testid="stSidebar"] {
    background: #111c30;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.title("⚗️ VLE Thermodynamics Laboratory")
st.caption("ASSIGNMENT 3  /  QUESTION 12.13  /  WILSON MODEL")

st.markdown("""
Calculate and visualize the temperature-composition (t-x-y) diagram
for a binary liquid mixture at constant pressure using the Wilson
activity coefficient model.
""")

st.divider()

# ---------------- SIDEBAR INPUTS ----------------
st.sidebar.title("⚙️ Calculation Settings")

st.sidebar.subheader("System Selection")
system = st.sidebar.selectbox("Binary system", ["Acetone – Water"])

st.sidebar.subheader("Operating Conditions")
P = st.sidebar.number_input(
    "Pressure (kPa)", min_value=10.0, max_value=500.0, value=101.33, step=1.0
)

st.sidebar.subheader("Antoine Constants")

st.sidebar.markdown("**Component 1: Acetone**")
A1 = st.sidebar.number_input("A1", value=14.3145, format="%.5f")
B1 = st.sidebar.number_input("B1", value=2756.22)
C1 = st.sidebar.number_input("C1", value=228.060)

st.sidebar.markdown("**Component 2: Water**")
A2 = st.sidebar.number_input("A2", value=16.3872, format="%.5f")
B2 = st.sidebar.number_input("B2", value=3885.70)
C2 = st.sidebar.number_input("C2", value=230.170)

st.sidebar.subheader("Wilson Parameters")
V1 = st.sidebar.number_input("V1 (cm³/mol)", value=74.05)
V2 = st.sidebar.number_input("V2 (cm³/mol)", value=18.07)
a12 = st.sidebar.number_input("a12 (cal/mol)", value=291.27)
a21 = st.sidebar.number_input("a21 (cal/mol)", value=1448.01)

st.sidebar.subheader("Graph Resolution")
n_points = st.sidebar.slider(
    "Number of calculation points", min_value=21, max_value=201, value=101, step=10
)

# ---------------- CONSTANTS ----------------
R = 1.987204  # cal/(mol K)

# ---------------- ANTOINE EQUATION ----------------
def antoine(A, B, C, temp_C):
    return math.exp(A - B / (temp_C + C))

def antoine_T(A, B, C, Pressure):
    """Boiling temperature (°C) of a pure component at a given pressure."""
    return B / (A - math.log(Pressure)) - C

# ---------------- WILSON EQUATION ----------------
def wilson(x1, T):
    x2 = 1.0 - x1

    L12 = (V2 / V1) * math.exp(-a12 / (R * T))
    L21 = (V1 / V2) * math.exp(-a21 / (R * T))

    term = L12 / (x1 + L12 * x2) - L21 / (x2 + L21 * x1)

    ln_gamma1 = -math.log(x1 + L12 * x2) + x2 * term
    ln_gamma2 = -math.log(x2 + L21 * x1) - x1 * term

    return math.exp(ln_gamma1), math.exp(ln_gamma2)

# ---------------- BUBBLE TEMPERATURE (BISECTION) ----------------
def bubble_T(x1, Pressure):
    """Find T (°C) such that sum(xi*gamma_i*Psat_i) = P."""
    x2 = 1.0 - x1

    def f(T_C):
        g1, g2 = wilson(x1, T_C + 273.15)
        return (x1 * g1 * antoine(A1, B1, C1, T_C)
                + x2 * g2 * antoine(A2, B2, C2, T_C)) - Pressure

    low, high = 0.0, 200.0
    for _ in range(100):
        mid = 0.5 * (low + high)
        if f(low) * f(mid) <= 0:
            high = mid
        else:
            low = mid
    return 0.5 * (low + high)

# ---------------- PURE COMPONENT BOILING POINTS ----------------
Tb1 = antoine_T(A1, B1, C1, P)
Tb2 = antoine_T(A2, B2, C2, P)

# ---------------- CALCULATIONS ----------------
x_values = np.linspace(0.0001, 0.9999, n_points)

bubble_temps = []
vapor_compositions = []
gamma1_values = []
gamma2_values = []

for x1 in x_values:
    T_C = bubble_T(x1, P)
    g1, g2 = wilson(x1, T_C + 273.15)
    y1 = x1 * g1 * antoine(A1, B1, C1, T_C) / P

    bubble_temps.append(T_C)
    vapor_compositions.append(y1)
    gamma1_values.append(g1)
    gamma2_values.append(g2)

# ---------------- DATAFRAME ----------------
df = pd.DataFrame({
    "Liquid mole fraction x1": x_values,
    "Vapor mole fraction y1": vapor_compositions,
    "Temperature (°C)": bubble_temps,
    "Temperature (K)": np.array(bubble_temps) + 273.15,
    "Gamma 1": gamma1_values,
    "Gamma 2": gamma2_values
})

# ---------------- METRICS ----------------
st.subheader("01 | System Overview")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Pressure", f"{P:.2f} kPa")
with c2:
    st.metric("Acetone boiling point", f"{Tb1:.2f} °C")
with c3:
    st.metric("Water boiling point", f"{Tb2:.2f} °C")
with c4:
    st.metric("Calculation points", n_points)

st.divider()

# ---------------- T-X-Y GRAPH ----------------
st.subheader("02 | Temperature – Composition Diagram")

plot_df = pd.concat([
    pd.DataFrame({
        "Mole fraction of acetone": x_values,
        "Temperature (°C)": bubble_temps,
        "Curve": "Bubble Point (T-x1)"
    }),
    pd.DataFrame({
        "Mole fraction of acetone": vapor_compositions,
        "Temperature (°C)": bubble_temps,
        "Curve": "Dew Point (T-y1)"
    })
], ignore_index=True)

fig = px.line(
    plot_df,
    x="Mole fraction of acetone",
    y="Temperature (°C)",
    color="Curve",
    color_discrete_map={
        "Bubble Point (T-x1)": "#38bdf8",
        "Dew Point (T-y1)": "#fb923c"
    },
    template="plotly_dark"
)

fig.update_traces(line=dict(width=3))
fig.update_layout(
    height=600,
    legend=dict(orientation="h", y=1.12, x=0, title_text=""),
    paper_bgcolor="#0b1220",
    plot_bgcolor="#111c30",
    margin=dict(l=40, r=30, t=50, b=50)
)
fig.update_xaxes(range=[0, 1], gridcolor="#293b56")
fig.update_yaxes(gridcolor="#293b56")

st.plotly_chart(fig, use_container_width=True)

st.caption("Blue: Bubble temperature curve | Orange: Dew temperature curve")

st.divider()

# ---------------- WILSON PARAMETERS ----------------
st.subheader("03 | Wilson Parameters (at mid-range temperature)")

T_mid = 0.5 * (Tb1 + Tb2) + 273.15

col1, col2 = st.columns(2)
with col1:
    st.metric("Wilson Λ12", f"{(V2/V1)*math.exp(-a12/(R*T_mid)):.5f}")
with col2:
    st.metric("Wilson Λ21", f"{(V1/V2)*math.exp(-a21/(R*T_mid)):.5f}")

# ---------------- RESULTS TABLE ----------------
st.subheader("04 | Calculation Results")

st.dataframe(df.round(5), use_container_width=True, height=350)

csv = df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Download Results CSV",
    data=csv,
    file_name="Q12_13_Acetone_Water.csv",
    mime="text/csv"
)

# ---------------- FORMULAS ----------------
with st.expander("View equations and calculation methodology"):
    st.markdown("""
    **Antoine equation**

    ln(Psat/kPa) = A - B/(T°C + C)

    **Modified Raoult's law (bubble point)**

    P = x1 γ1 P1sat(T) + x2 γ2 P2sat(T)

    For each x1, the temperature T is found by bisection so that the
    sum equals the specified pressure P. The activity coefficients are
    recomputed at every trial temperature.

    **Vapor composition**

    y1 = x1 γ1 P1sat / P

    **Wilson equation**

    Activity coefficients are calculated using the binary Wilson model.

    **Gas constant**

    R = 1.987204 cal/(mol K)
    """)

st.divider()

st.markdown("""
<div style="text-align:center;color:#94a3b8;">
VLE Thermodynamics Laboratory | Assignment 3 | Question 12.13
<br>
Wilson Equation | Acetone – Water
</div>
""", unsafe_allow_html=True)
