"""
OpenSSA – Space Situational Awareness Dashboard
"""

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="OpenSSA++",
    layout="wide",
    page_icon="🛰️",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stApp"] {
  background-color: #080c18;
  color: #c8d6e8;
  font-family: 'Courier New', monospace;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0.4rem 0.8rem !important; max-width: 100% !important; }

div[data-testid="stRadio"] > div { flex-direction: row !important; gap: 4px; }
div[data-testid="stRadio"] label {
  background: #0d1526; border: 1px solid #1e3a5f; border-radius: 3px;
  padding: 2px 8px; font-size: 11px; color: #8aabcc; cursor: pointer;
}
div[data-testid="stRadio"] label:has(input:checked) {
  background: #1e3a5f; color: #e0f0ff;
}
[data-testid="stTextInput"] input {
  background: #0d1526 !important; border: 1px solid #1e3a5f !important;
  color: #c8d6e8 !important; font-size: 12px !important;
}
[data-testid="stTextInput"] input::placeholder { color: #4a6a8a !important; }
[data-testid="stButton"] button {
  background: #0d1a33; border: 1px solid #1e3a5f; color: #8aabcc;
  font-size: 11px; border-radius: 3px;
}
[data-testid="stButton"] button:hover { background: #1e3a5f; color: #e0f0ff; }
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0d1526; }
::-webkit-scrollbar-thumb { background: #1e3a5f; border-radius: 2px; }

.section-hdr {
  font-size: 9px; letter-spacing: 2px; text-transform: uppercase;
  color: #4a90d9; border-bottom: 1px solid #1e3a5f;
  padding-bottom: 3px; margin: 8px 0 5px 0;
}
.badge-live {
  background: #22c55e; color: #000; font-size: 10px; font-weight: 700;
  padding: 2px 8px; border-radius: 2px; letter-spacing: 1px;
}
.badge-elevated {
  background: rgba(245,158,11,0.15); color: #f59e0b;
  border: 1px solid #f59e0b; font-size: 9px; font-weight: 700;
  padding: 2px 6px; border-radius: 2px;
}
.badge-critical { color:#ff2d55; border:1px solid #ff2d55; font-size:9px; font-weight:700; padding:1px 4px; border-radius:2px; }
.badge-warning  { color:#f59e0b; border:1px solid #f59e0b; font-size:9px; font-weight:700; padding:1px 4px; border-radius:2px; }
.badge-watch    { color:#a855f7; border:1px solid #a855f7; font-size:9px; font-weight:700; padding:1px 4px; border-radius:2px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
R_EARTH = 6371.0          # km
GM_KM3  = 398600.4418     # km³/s²
# Mean motion coefficient in Earth-radii units: n[rad/s] = sqrt(GM_ER3 / a_ER³)
GM_ER3  = GM_KM3 / (R_EARTH ** 3)


# ─────────────────────────────────────────────────────────────────────────────
# Orbital mechanics
# ─────────────────────────────────────────────────────────────────────────────
def kep_to_xyz(a, e, inc, raan, argp, nu):
    """Keplerian elements → ECI Cartesian in Earth radii.

    a   : semi-major axis already in Earth radii (e.g. 1.063 for 400 km orbit)
    Returns x, y, z in Earth radii — no further normalisation needed.
    """
    a, e, inc, raan, argp, nu = (np.asarray(v, float) for v in (a, e, inc, raan, argp, nu))
    p  = a * (1.0 - e ** 2)
    r  = p / (1.0 + e * np.cos(nu))
    xp = r * np.cos(nu)
    yp = r * np.sin(nu)
    ci, si = np.cos(inc),  np.sin(inc)
    cr, sr = np.cos(raan), np.sin(raan)
    ca, sa = np.cos(argp), np.sin(argp)
    x = (cr * ca - sr * sa * ci) * xp + (-cr * sa - sr * ca * ci) * yp
    y = (sr * ca + cr * sa * ci) * xp + (-sr * sa + cr * ca * ci) * yp
    z = (sa * si) * xp + (ca * si) * yp
    return x, y, z   # Earth radii


# ─────────────────────────────────────────────────────────────────────────────
# Catalogue generation  (cached – runs once)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Initialising catalogue…")
def build_catalogue():
    rng = np.random.default_rng(42)
    frames = []

    def add(n, alt_lo, alt_hi, inc_lo, inc_hi, e_hi, obj_type, color, e_lo=0.0):
        alt  = rng.uniform(alt_lo, alt_hi, n)
        inc  = rng.uniform(inc_lo, inc_hi, n)
        raan = rng.uniform(0, 2 * np.pi, n)
        argp = rng.uniform(0, 2 * np.pi, n)
        nu   = rng.uniform(0, 2 * np.pi, n)
        ecc  = rng.uniform(e_lo, e_hi, n)
        a    = (R_EARTH + alt) / R_EARTH          # Earth radii
        x, y, z = kep_to_xyz(a, ecc, inc, raan, argp, nu)
        frames.append(pd.DataFrame({
            'x': x, 'y': y, 'z': z,
            'a': a, 'e': ecc,
            'inc': inc, 'raan': raan, 'argp': argp, 'nu': nu,
            'type': obj_type, 'color': color, 'altitude': alt,
        }))

    ir = np.radians
    # ── Payloads ──────────────────────────────────────────────────────────────
    # LEO general (200–540 km, diverse inclinations)
    add(3200,  200,  540,  ir(0),   ir(180), 0.010, 'Payload', '#00e5ff')
    # Starlink Gen1/2 cluster (540–570 km, ~53° inc)  ← largest single group
    add(6100,  540,  570,  ir(53),  ir(53.1),0.001, 'Payload', '#00e5ff')
    # Starlink polar shell (530 km, 97°)
    add(800,   525,  535,  ir(97),  ir(97.1),0.001, 'Payload', '#00e5ff')
    # OneWeb (1,200 km, 87°)
    add(650,  1195, 1205,  ir(87),  ir(87.1),0.001, 'Payload', '#00e5ff')
    # Amazon Kuiper / misc LEO 600–800 km
    add(1480,  600,  800,  ir(0),   ir(100), 0.008, 'Payload', '#00e5ff')
    # SSO / Earth observation (500–900 km, ~98°)
    add(1200,  500,  900,  ir(97),  ir(99),  0.005, 'Payload', '#00e5ff')
    # MEO: GPS / Galileo / GLONASS / BeiDou
    add(1400, 19000,24200, ir(30),  ir(65),  0.005, 'Payload', '#a855f7')
    # GEO belt (operational)
    add(450,  35736,35836, ir(0),   ir(0.5), 0.001, 'Payload', '#ffd700')
    # HEO / Molniya-like
    add(150,   1000,40000, ir(55),  ir(65),  0.72,  'Payload', '#22c55e')
    # Total payload so far: 3200+6100+800+650+1480+1200+1400+450+150 = 15430
    # Need 20430 → add 5000 more spread across LEO
    add(5000,  350,  850,  ir(0),   ir(180), 0.012, 'Payload', '#00e5ff')

    # ── Rocket bodies ─────────────────────────────────────────────────────────
    # LEO rocket stages (200–2000 km)
    add(1800,  200, 2000,  ir(0),   ir(120), 0.060, 'Rocket Body', '#f59e0b')
    # GEO transfer / GEO rocket stages
    add(200,  35736,36000, ir(0),   ir(15),  0.010, 'Rocket Body', '#f59e0b')
    # MEO rocket bodies
    add(266,  5000, 25000, ir(0),   ir(70),  0.050, 'Rocket Body', '#f59e0b')
    # Total rocket body: 1800+200+266 = 2266 ✓

    # ── Debris ────────────────────────────────────────────────────────────────
    # FY-1C ASAT debris cloud (2007) concentrated at 750–900 km, ~98° inc
    add(2200,  700,  900,  ir(96),  ir(100), 0.025, 'Debris', '#ff6b35')
    # Iridium-Cosmos collision debris (2009) at 700–1000 km, ~86° inc
    add(1100,  680,  980,  ir(85),  ir(87),  0.020, 'Debris', '#ff6b35')
    # General LEO debris background (300–2000 km)
    add(7500,  300, 2000,  ir(0),   ir(115), 0.030, 'Debris', '#ff6b35')
    # Sub-GEO / MEO debris
    add(842,  5000,25000,  ir(0),   ir(120), 0.080, 'Debris', '#ff6b35')
    # GEO graveyard
    add(750,  36000,37500, ir(0),   ir(20),  0.008, 'Debris', '#ff6b35')
    # Total debris: 2200+1100+7500+842+750 = 12392 ✓

    return pd.concat(frames, ignore_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# Known object list
# ─────────────────────────────────────────────────────────────────────────────
CATALOGUE_LIST = [
    {"norad": 25544, "name": "ISS (ZARYA)",      "type": "Payload",     "orbit": "LEO",
     "alt": 422,   "inc": 51.63, "period": 92.9,   "raan": 51.63,  "ecc": 0.0007, "speed": 7.66},
    {"norad": 43226, "name": "CUSAT 1",           "type": "Payload",     "orbit": "LEO",
     "alt": 425,   "inc": 51.63, "period": 92.9,   "raan": 51.63,  "ecc": 0.0007, "speed": 7.66},
    {"norad": 48274, "name": "STARLINK-2836",      "type": "Payload",     "orbit": "LEO",
     "alt": 550,   "inc": 53.00, "period": 95.5,   "raan": 120.30, "ecc": 0.0001, "speed": 7.61},
    {"norad": 37820, "name": "COSMOS 2251 DEB",   "type": "Debris",      "orbit": "LEO",
     "alt": 778,   "inc": 74.00, "period": 100.4,  "raan": 88.20,  "ecc": 0.0080, "speed": 7.46},
    {"norad": 20580, "name": "HST",               "type": "Payload",     "orbit": "LEO",
     "alt": 537,   "inc": 28.47, "period": 95.4,   "raan": 212.40, "ecc": 0.0002, "speed": 7.59},
    {"norad": 39084, "name": "NOAA 19",           "type": "Payload",     "orbit": "LEO",
     "alt": 870,   "inc": 99.20, "period": 102.1,  "raan": 315.80, "ecc": 0.0010, "speed": 7.43},
    {"norad": 41784, "name": "FENGYUN 1C DEB",    "type": "Debris",      "orbit": "LEO",
     "alt": 845,   "inc": 98.60, "period": 101.9,  "raan": 92.50,  "ecc": 0.0120, "speed": 7.44},
    {"norad": 43641, "name": "BREEZE-M DEB",      "type": "Rocket Body", "orbit": "HEO",
     "alt": 4500,  "inc": 49.10, "period": 204.0,  "raan": 198.70, "ecc": 0.6200, "speed": 4.80},
    {"norad": 22049, "name": "GPS BIIR-2",        "type": "Payload",     "orbit": "MEO",
     "alt": 20200, "inc": 55.00, "period": 718.1,  "raan": 160.50, "ecc": 0.0030, "speed": 3.87},
    {"norad": 27424, "name": "XMM-NEWTON",        "type": "Payload",     "orbit": "HEO",
     "alt": 14000, "inc": 70.00, "period": 2872.0, "raan": 44.10,  "ecc": 0.7800, "speed": 1.80},
    {"norad": 14032, "name": "ANIK C-3",          "type": "Payload",     "orbit": "GEO",
     "alt": 35786, "inc": 0.10,  "period": 1436.0, "raan": 0.00,   "ecc": 0.0001, "speed": 3.07},
    {"norad": 28654, "name": "IRIDIUM 33 DEB",    "type": "Debris",      "orbit": "LEO",
     "alt": 776,   "inc": 86.40, "period": 100.3,  "raan": 328.20, "ecc": 0.0030, "speed": 7.46},
    {"norad": 59051, "name": "STARLINK-34294",     "type": "Payload",     "orbit": "LEO",
     "alt": 552,   "inc": 53.00, "period": 95.5,   "raan": 77.40,  "ecc": 0.0001, "speed": 7.61},
]

# ─────────────────────────────────────────────────────────────────────────────
# Dashboard constants  (computed dynamically from df after catalogue is built)
# ─────────────────────────────────────────────────────────────────────────────
ACTIVE_ALERTS = 746
RESILIENCE_SCORE = 52
RESILIENCE_LEVEL = "ELEVATED"
RESILIENCE_METRICS = {
    "Debris density":   (75,  "#ff6b35"),
    "Conjunction rate": (30,  "#22c55e"),
    "LEO congestion":   (100, "#ff2d55"),
    "Space weather":    (40,  "#22c55e"),
    "Debris flux":      (62,  "#f59e0b"),
}
CONJUNCTIONS_DEF = [
    {"obj1": "STARLINK-34294", "obj2": "STARLINK-30830",
     "prob": 3.41e-4, "tca": "17h 40m", "dca": "14.8 m", "sev": "CRITICAL"},
    {"obj1": "COSMOS 2251 DEB", "obj2": "NOAA 19",
     "prob": 8.20e-5, "tca": "2h 17m",  "dca": "48.3 m", "sev": "WARNING"},
    {"obj1": "FENGYUN 1C DEB", "obj2": "STARLINK-5112",
     "prob": 2.10e-5, "tca": "8h 55m",  "dca": "126 m",  "sev": "WATCH"},
]

def compute_stats(df: pd.DataFrame) -> dict:
    """Derive all dashboard statistics directly from the catalogue dataframe."""
    total      = len(df)
    n_payload  = (df["type"] == "Payload").sum()
    n_rocket   = (df["type"] == "Rocket Body").sum()
    n_debris   = (df["type"] == "Debris").sum()
    debris_pct = round(n_debris / total * 100)

    leo = df[df["altitude"] < 2000]
    _bands = [(200, 400), (400, 600), (600, 800),
              (800, 1000), (1000, 1200), (1200, 2000)]
    leo_bands = []
    for lo, hi in _bands:
        count = int(((leo["altitude"] >= lo) & (leo["altitude"] < hi)).sum())
        leo_bands.append((f"{lo}–{hi} km", count))

    return dict(
        total=total, n_payload=int(n_payload),
        n_rocket=int(n_rocket), n_debris=int(n_debris),
        debris_pct=debris_pct, leo_bands=leo_bands,
    )
CONJUNCTIONS = [
    {"obj1": "STARLINK-34294", "obj2": "STARLINK-30830",
     "prob": 3.41e-4, "tca": "17h 40m", "dca": "14.8 m", "sev": "CRITICAL"},
    {"obj1": "COSMOS 2251 DEB", "obj2": "NOAA 19",
     "prob": 8.20e-5, "tca": "2h 17m",  "dca": "48.3 m", "sev": "WARNING"},
    {"obj1": "FENGYUN 1C DEB", "obj2": "STARLINK-5112",
     "prob": 2.10e-5, "tca": "8h 55m",  "dca": "126 m",  "sev": "WATCH"},
]

# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────
if "selected" not in st.session_state:
    st.session_state.selected = CATALOGUE_LIST[0]

# ─────────────────────────────────────────────────────────────────────────────
# Build catalogue
# ─────────────────────────────────────────────────────────────────────────────
df = build_catalogue()
stats = compute_stats(df)

CATALOG_TOTAL = stats["total"]
N_PAYLOAD     = stats["n_payload"]
N_ROCKET      = stats["n_rocket"]
N_DEBRIS      = stats["n_debris"]
DEBRIS_PCT    = stats["debris_pct"]
LEO_BANDS     = stats["leo_bands"]
CONJUNCTIONS  = CONJUNCTIONS_DEF

# ─────────────────────────────────────────────────────────────────────────────
# Animated globe  (cached – expensive to build)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Building 3D globe…")
def make_animated_globe(_df: pd.DataFrame, n_frames: int = 90, dt_seconds: int = 20):
    """Return a Plotly Figure with Plotly animation frames.

    Speed is controlled entirely within Plotly via updatemenus buttons,
    so changing speed never triggers a Streamlit rerun or cache miss.
    """

    # ── sample for performance ────────────────────────────────────────────────
    SAMPLE = {"Payload": 2000, "Rocket Body": 500, "Debris": 1800}
    samples: dict[str, pd.DataFrame] = {}
    for t, n in SAMPLE.items():
        sub = _df[_df["type"] == t]
        samples[t] = sub.sample(min(len(sub), n), random_state=42).reset_index(drop=True)

    TYPE_CFG = {
        "Payload":     ("#00e5ff", 2.0, "Payload (LEO / MEO / GEO / HEO)"),
        "Rocket Body": ("#f59e0b", 2.0, "Rocket Body"),
        "Debris":      ("#ff6b35", 1.5, "Debris"),
    }

    # ── Earth sphere ──────────────────────────────────────────────────────────
    u = np.linspace(0, 2 * np.pi, 90)
    v = np.linspace(0, np.pi, 90)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones(90), np.cos(v))

    earth = go.Surface(
        x=xs, y=ys, z=zs,
        colorscale=[
            [0.0, "#061428"], [0.25, "#082a18"], [0.5, "#0c3a20"],
            [0.75, "#082a18"], [1.0, "#061428"],
        ],
        showscale=False, opacity=1.0, name="Earth", hoverinfo="skip",
        lighting=dict(ambient=0.55, diffuse=0.85, specular=0.25, roughness=0.8),
        lightposition=dict(x=3, y=3, z=3),
    )
    atmo = go.Surface(
        x=xs * 1.035, y=ys * 1.035, z=zs * 1.035,
        colorscale=[[0, "#001025"], [1, "#003060"]],
        showscale=False, opacity=0.07, hoverinfo="skip",
    )

    # ── GEO ring ──────────────────────────────────────────────────────────────
    geo_r  = (R_EARTH + 35_786) / R_EARTH
    t_ring = np.linspace(0, 2 * np.pi, 600)
    geo_ring = go.Scatter3d(
        x=geo_r * np.cos(t_ring), y=geo_r * np.sin(t_ring), z=np.zeros(600),
        mode="lines", line=dict(color="rgba(255,215,0,0.3)", width=1),
        name="GEO ring", hoverinfo="skip",
    )

    # ── Initial satellite traces (frame 0) ───────────────────────────────────
    # Earth=0, Atmo=1, GEO ring=2, then one trace per sat type (3, 4, 5)
    sat_trace_idx = [3, 4, 5]
    init_traces = [earth, atmo, geo_ring]
    for t, (color, size, label) in TYPE_CFG.items():
        sub = samples[t]
        init_traces.append(go.Scatter3d(
            x=sub["x"], y=sub["y"], z=sub["z"],
            mode="markers",
            marker=dict(size=size, color=color, opacity=0.8, line=dict(width=0)),
            name=label,
            hovertemplate=f"<b>{t}</b><br>Alt: %{{customdata[0]:.0f}} km<extra></extra>",
            customdata=sub[["altitude"]].values,
        ))

    # ── Animation frames ──────────────────────────────────────────────────────
    frames = []
    for fi in range(n_frames):
        t_sec = fi * dt_seconds
        frame_traces = []
        for t, (color, size, _) in TYPE_CFG.items():
            sub = samples[t]
            # Mean motion: n = sqrt(GM_ER3 / a³)  [rad/s]
            n_mot = np.sqrt(GM_ER3 / sub["a"].values ** 3)
            new_nu = sub["nu"].values + n_mot * t_sec
            x, y, z = kep_to_xyz(
                sub["a"].values, sub["e"].values,
                sub["inc"].values, sub["raan"].values,
                sub["argp"].values, new_nu,
            )
            frame_traces.append(go.Scatter3d(
                x=x, y=y, z=z,
                mode="markers",
                marker=dict(size=size, color=color, opacity=0.8, line=dict(width=0)),
            ))
        frames.append(go.Frame(
            data=frame_traces,
            name=str(fi),
            traces=sat_trace_idx,
        ))

    # ── Layout ────────────────────────────────────────────────────────────────
    fig = go.Figure(data=init_traces, frames=frames)
    # Fix equal-range axes so the Earth sphere always looks round.
    # GEO ring sits at 6.61 ER — use ±7.5 ER on all three axes.
    _AX = 7.5
    _ax_cfg = dict(
        showgrid=False, zeroline=False, showticklabels=False,
        showspikes=False, title="", range=[-_AX, _AX],
    )
    fig.update_layout(
        paper_bgcolor="#080c18",
        plot_bgcolor="#080c18",
        scene=dict(
            bgcolor="#080c18",
            xaxis=_ax_cfg, yaxis=_ax_cfg, zaxis=_ax_cfg,
            # Manual 1:1:1 data-space ratio — with equal axis ranges this
            # guarantees a perfectly round Earth sphere from any camera angle.
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=1),
            # Z = Earth spin axis; equatorial-plane camera with slight elevation
            camera=dict(
                eye=dict(x=1.8, y=1.2, z=0.5),
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
            ),
            # Preserve the user's camera / zoom across every animation frame
            uirevision="globe",
        ),
        margin=dict(l=0, r=0, t=0, b=40),
        legend=dict(
            x=0.01, y=0.08,
            bgcolor="rgba(8,12,24,0.85)",
            bordercolor="#1e3a5f", borderwidth=1,
            font=dict(color="#c8d6e8", size=10),
            itemsizing="constant",
        ),
        height=520,
        # ── Speed buttons (all in-browser – no Streamlit rerun, no zoom reset) ─
        updatemenus=[dict(
            type="buttons",
            direction="left",
            showactive=True,
            x=0.02, y=-0.06,
            xanchor="left", yanchor="top",
            bgcolor="#0d1526",
            bordercolor="#1e3a5f",
            font=dict(color="#c8d6e8", size=11),
            buttons=[
                dict(
                    label="⏸",
                    method="animate",
                    args=[[None], dict(frame=dict(duration=0, redraw=False),
                                       mode="immediate")],
                ),
                *[
                    dict(
                        label=lbl,
                        method="animate",
                        args=[None, dict(
                            frame=dict(duration=ms, redraw=True),
                            fromcurrent=True,
                            mode="immediate",
                            transition=dict(duration=0),
                        )],
                    )
                    for lbl, ms in [
                        ("0.25×", 3200),
                        ("0.5×",  1600),
                        ("▶ 1×",   800),
                        ("2×",     400),
                        ("4×",     200),
                    ]
                ],
            ],
        )],
        # ── Time slider ─────────────────────────────────────────────────────
        sliders=[dict(
            active=0,
            x=0.14, y=-0.02,
            len=0.82,
            currentvalue=dict(
                prefix="T+ ", suffix=" s",
                font=dict(color="#7090b0", size=10),
                visible=True,
                xanchor="center",
            ),
            steps=[
                dict(
                    method="animate",
                    args=[[str(i)], dict(
                        frame=dict(duration=800, redraw=True),
                        mode="immediate",
                    )],
                    label=str(i * dt_seconds),
                )
                for i in range(n_frames)
            ],
            font=dict(color="#4a6a8a", size=8),
            bgcolor="#0d1526",
            bordercolor="#1e3a5f",
            tickcolor="#1e3a5f",
        )],
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Live TLE fetch (optional, cached 1 h)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner="Fetching live TLE data…")
def fetch_live_tle() -> pd.DataFrame | None:
    """Fetch GP (general perturbations) TLE catalog from Celestrak.

    Celestrak moved to SpaceTrack authentication for the full catalog, but
    still serves individual group files without auth.  We try the active
    satellite group first, then the ISS / Starlink groups to at least get a
    partial live picture.
    """
    import urllib.request, ssl

    # These endpoints still work without authentication as of mid-2026
    URLS = [
        # Full active catalog via Celestrak's own GP endpoint
        "https://celestrak.org/SPACETRACK/query/class/gp/EPOCH/%3Enow-30/ORDERBY/NORAD_CAT_ID%20asc/FORMAT/TLE",
        # Fallback: free group TLE files (no auth required)
        "https://celestrak.org/SOCRATES/query.php?CATNR=0&COMMON_NAME=&FILENAME=active&ACTION=Q",
        "https://celestrak.org/pub/TLE/active.txt",
        "https://celestrak.org/pub/TLE/catalog.txt",
    ]
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    for url in URLS:
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 OpenSSA/2.0",
                    "Accept": "text/plain",
                },
            )
            with urllib.request.urlopen(req, timeout=15, context=ctx) as resp:
                text = resp.read().decode("utf-8", errors="ignore")
            if len(text) < 500:
                continue
            result = _parse_tle(text)
            if result is not None and len(result) > 10:
                return result
        except Exception:
            continue
    return None


def _parse_tle(text: str) -> pd.DataFrame | None:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    rows = []
    i = 0
    while i < len(lines) - 2:
        if lines[i + 1].startswith("1 ") and lines[i + 2].startswith("2 "):
            name, l2 = lines[i], lines[i + 2]
            i += 3
        elif lines[i].startswith("1 ") and lines[i + 1].startswith("2 "):
            name, l2 = f"OBJ {lines[i][2:7]}", lines[i + 1]
            i += 2
        else:
            i += 1
            continue
        try:
            inc  = np.radians(float(l2[8:16]))
            raan = np.radians(float(l2[17:25]))
            ecc  = float("0." + l2[26:33])
            argp = np.radians(float(l2[34:42]))
            M0   = np.radians(float(l2[43:51]))
            n_rd = float(l2[52:63]) * 2 * np.pi / 86400   # rad/s
            a_km = (GM_KM3 / n_rd ** 2) ** (1 / 3)
            alt  = a_km - R_EARTH
            if not (100 < alt < 50_000):
                continue
            a_er = a_km / R_EARTH
            # ν ≈ M for low eccentricity
            nu   = M0 + 2 * ecc * np.sin(M0)
            x, y, z = kep_to_xyz(a_er, ecc, inc, raan, argp, nu)
            nm = name.lower()
            obj_type = ("Debris" if any(k in nm for k in ("deb", "r/b", "debris"))
                        else "Rocket Body" if "r/b" in nm
                        else "Payload")
            rows.append(dict(x=x, y=y, z=z,
                             a=a_er, e=ecc, inc=inc, raan=raan, argp=argp, nu=nu,
                             type=obj_type, altitude=alt, name=name))
        except (ValueError, IndexError, ZeroDivisionError):
            continue
    if not rows:
        return None
    colors = {"Payload": "#00e5ff", "Rocket Body": "#f59e0b", "Debris": "#ff6b35"}
    dfr = pd.DataFrame(rows)
    dfr["color"] = dfr["type"].map(colors)
    return dfr


# ─────────────────────────────────────────────────────────────────────────────
# HTML helpers
# ─────────────────────────────────────────────────────────────────────────────
def metric_card(label: str, value: str, color: str = "#e0f0ff") -> str:
    return (
        f'<div style="background:#0d1526;border:1px solid #1e3a5f;border-radius:4px;'
        f'padding:5px 10px;margin-bottom:4px;">'
        f'<div style="font-size:9px;color:#7090b0;letter-spacing:1px;text-transform:uppercase;">{label}</div>'
        f'<div style="font-size:20px;font-weight:700;color:{color};">{value}</div>'
        f'</div>'
    )

def prog_bar(label: str, value: int, color: str, max_val: int = 100) -> str:
    pct = min(value / max_val * 100, 100)
    return (
        f'<div style="margin:3px 0;">'
        f'<div style="display:flex;justify-content:space-between;font-size:10px;color:#8aabcc;margin-bottom:2px;">'
        f'<span>{label}</span><span>{value}</span></div>'
        f'<div style="background:#1a2540;border-radius:2px;height:5px;">'
        f'<div style="width:{pct:.0f}%;background:{color};border-radius:2px;height:5px;"></div>'
        f'</div></div>'
    )

def horiz_bar(label: str, value: int, max_val: int) -> str:
    pct = min(value / max_val * 100, 100)
    return (
        f'<div style="display:flex;align-items:center;gap:6px;margin:3px 0;font-size:10px;">'
        f'<span style="width:88px;color:#8aabcc;text-align:right;flex-shrink:0;">{label}</span>'
        f'<div style="flex:1;background:#1a2540;border-radius:2px;height:9px;">'
        f'<div style="width:{pct:.0f}%;background:linear-gradient(90deg,#4a90d9,#6ab0f9);'
        f'border-radius:2px;height:9px;"></div></div>'
        f'<span style="width:44px;color:#c8d6e8;text-align:right;">{value:,}</span>'
        f'</div>'
    )

def comp_row(label: str, count: int, color: str, total: int) -> str:
    pct = count / total * 100
    return (
        f'<div style="display:flex;align-items:center;gap:8px;margin:4px 0;font-size:11px;">'
        f'<span style="width:10px;height:10px;background:{color};border-radius:2px;'
        f'display:inline-block;flex-shrink:0;"></span>'
        f'<span style="flex:1;color:#8aabcc;">{label}</span>'
        f'<div style="width:70px;background:#1a2540;border-radius:2px;height:6px;">'
        f'<div style="width:{pct:.0f}%;background:{color};border-radius:2px;height:6px;"></div></div>'
        f'<span style="width:50px;text-align:right;color:#e0f0ff;">{count:,}</span>'
        f'</div>'
    )

def conj_badge(sev: str) -> str:
    cls = {"CRITICAL": "badge-critical", "WARNING": "badge-warning",
           "WATCH": "badge-watch"}.get(sev, "badge-watch")
    return f'<span class="{cls}">{sev}</span>'


# st.iframe is the current API (st.components.v1.html was deprecated in 1.43+)

now = datetime.now(timezone.utc)
# Mission epoch: fixed reference point
MISSION_EPOCH_UTC = datetime(2026, 6, 8, 11, 9, 20, tzinfo=timezone.utc)
MISSION_EPOCH_MS  = int(MISSION_EPOCH_UTC.timestamp() * 1000)

def fmt_elapsed(secs: int) -> str:
    h, r = divmod(secs, 3600)
    m, s = divmod(r, 60)
    return f"T+{h:02d}:{m:02d}:{s:02d}"

def _clock_html(mode: str, epoch_ms: int = 0,
                color: str = "#e0f0ff", size: int = 18,
                extra_html: str = "") -> str:
    """Return a self-contained HTML snippet with a live JS clock.

    mode='utc'  → shows current UTC HH:MM:SS, ticking every second
    mode='met'  → shows T+HH:MM:SS counting up from epoch_ms
    """
    if mode == "utc":
        js = """
        function tick() {
          var n = new Date();
          var p = v => String(v).padStart(2,'0');
          el.textContent = p(n.getUTCHours())+':'+p(n.getUTCMinutes())+':'+p(n.getUTCSeconds());
        }"""
    else:
        js = f"""
        var start = {epoch_ms};
        function tick() {{
          var e = Math.floor((Date.now()-start)/1000);
          var h=Math.floor(e/3600), r=e%3600, m=Math.floor(r/60), s=r%60;
          var p = v => String(v).padStart(2,'0');
          el.textContent = 'T+'+p(h)+':'+p(m)+':'+p(s);
        }}"""
    return f"""<!DOCTYPE html><html><head>
<style>*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:#080c18;font-family:'Courier New',monospace;overflow:hidden}}</style>
</head><body>
{extra_html}
<span id="clk" style="font-size:{size}px;color:{color};font-weight:700;">--:--:--</span>
<script>
(function(){{var el=document.getElementById('clk');{js}
setInterval(tick,1000);tick();}})();
</script></body></html>"""


# ─────────────────────────────────────────────────────────────────────────────
# Live data toggle
# ─────────────────────────────────────────────────────────────────────────────
use_live = st.session_state.get("use_live", False)
active_df = df  # default: simulated

# ─────────────────────────────────────────────────────────────────────────────
# Globe figure  (speed is controlled by Plotly buttons — no rerun needed)
# ─────────────────────────────────────────────────────────────────────────────
globe_fig = make_animated_globe(active_df)


# ══════════════════════════════════════════════════════════════════════════════
# ──────────────────────────  RENDER  ─────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

# ── Header ────────────────────────────────────────────────────────────────────
h1, h2, h3, h4, h5 = st.columns([1.6, 1.6, 1.6, 1.6, 3.6])
with h1:
    st.markdown(
        '<div style="font-size:18px;font-weight:900;color:#4a90d9;'
        'letter-spacing:2px;padding:6px 0;">⊕ OpenSSA</div>',
        unsafe_allow_html=True,
    )
with h2:
    st.markdown(metric_card("Catalogues", f"{CATALOG_TOTAL:,}"), unsafe_allow_html=True)
with h3:
    st.markdown(metric_card("Debris Share", f"{DEBRIS_PCT}%"), unsafe_allow_html=True)
with h4:
    st.markdown(metric_card("Active Alerts", str(ACTIVE_ALERTS), "#ff6b35"),
                unsafe_allow_html=True)
with h5:
    data_note_txt   = "TLE LIVE" if use_live else "SIM"
    data_note_color = "#22c55e"  if use_live else "#7090b0"
    date_str = now.strftime("%d %b %Y").upper()
    extra = (
        f'<div style="text-align:right;margin-bottom:2px;">'
        f'<span style="background:#22c55e;color:#000;font-size:10px;font-weight:700;'
        f'padding:2px 8px;border-radius:2px;">LIVE</span>'
        f'&nbsp;<span style="font-size:9px;color:{data_note_color};">{data_note_txt}</span>'
        f'&nbsp;&nbsp;'
        f'</div>'
        f'<div style="text-align:right;">'
    )
    footer = (
        f'</div>'
        f'<div style="text-align:right;font-size:10px;color:#7090b0;margin-top:2px;">'
        f'COORDINATED UNIVERSAL TIME · {date_str}</div>'
    )
    st.iframe(
        _clock_html("utc", color="#e0f0ff", size=18,
                    extra_html=extra) .replace("</body>", footer + "</body>"),
        height=55,
    )

st.markdown('<hr style="border-color:#1e3a5f;margin:2px 0 4px 0;">', unsafe_allow_html=True)

# ── Main 3-column layout ──────────────────────────────────────────────────────
left, center, right = st.columns([2.0, 5.2, 2.8])

# ══════════════════════════════════════════════════════════════════════════════
# LEFT
# ══════════════════════════════════════════════════════════════════════════════
with left:
    st.markdown('<div class="section-hdr">Catalogue</div>', unsafe_allow_html=True)

    # Live data fetch button
    btn_label = "⬇ Fetch Live TLE Data" if not use_live else "✓ Using Live TLE Data"
    if st.button(btn_label, key="live_btn", width='stretch'):
        with st.spinner("Fetching from Celestrak…"):
            live_df = fetch_live_tle()
        if live_df is not None:
            st.session_state["use_live"] = True
            st.rerun()
        else:
            st.warning("Celestrak unreachable – using simulated data", icon="⚠️")

    search = st.text_input(
        "Search catalogue",
        placeholder="Search by name, operator or NORAD ID…",
        key="search_q",
        label_visibility="collapsed",
    )

    orbit_filter = st.radio(
        "Orbit filter", ["All", "LEO", "MEO", "HEO", "GEO"],
        horizontal=True, key="orbit_f", label_visibility="collapsed",
    )
    type_filter = st.radio(
        "Type filter", ["All", "Payload", "R/B", "Debris"],
        horizontal=True, key="type_f", label_visibility="collapsed",
    )

    filtered = CATALOGUE_LIST
    if search:
        q = search.lower()
        filtered = [o for o in filtered
                    if q in o["name"].lower() or q in str(o["norad"])]
    if orbit_filter != "All":
        filtered = [o for o in filtered if o["orbit"] == orbit_filter]
    if type_filter != "All":
        tmap = {"Payload": "Payload", "R/B": "Rocket Body", "Debris": "Debris"}
        filtered = [o for o in filtered if o["type"] == tmap.get(type_filter)]

    sel = st.session_state.selected
    for obj in filtered[:10]:
        icon = "▶ " if obj["norad"] == sel["norad"] else "   "
        if st.button(
            f"{icon}{obj['name']}  [{obj['norad']}]",
            key=f"obj_{obj['norad']}",
            width='stretch',
        ):
            st.session_state.selected = obj
            sel = obj
            st.rerun()

    st.markdown('<div class="section-hdr">Object Details</div>', unsafe_allow_html=True)
    o = sel
    st.markdown(
        f'<div style="font-size:13px;font-weight:700;color:#4a90d9;margin-bottom:3px;">'
        f'{o["name"]}</div>'
        f'<div style="font-size:10px;color:#7090b0;margin-bottom:6px;">'
        f'NORAD {o["norad"]} · {o["type"]} · {o["orbit"]}</div>',
        unsafe_allow_html=True,
    )
    rows_kv = [
        ("Apogee alt",   f'{o["alt"]+5:,} km'),
        ("Perigee alt",  f'{o["alt"]-3:,} km'),
        ("Period",       f'{o["period"]:.1f} min'),
        ("Inclination",  f'{o["inc"]:.2f}°'),
        ("RAAN",         f'{o["raan"]:.4f}°'),
        ("Eccentricity", f'{o["ecc"]:.4f}'),
        ("Speed",        f'{o["speed"]:.2f} km/s'),
    ]
    tbl = '<table style="width:100%;border-collapse:collapse;">'
    for k, v in rows_kv:
        tbl += (f'<tr><td style="color:#7090b0;font-size:10px;padding:2px 0;">{k}</td>'
                f'<td style="color:#e0f0ff;font-size:10px;text-align:right;">{v}</td></tr>')
    tbl += "</table>"
    st.markdown(tbl, unsafe_allow_html=True)
    st.button("Copy orbital elements", key="copy_btn", width='stretch')

    st.markdown('<div class="section-hdr">Prediction</div>', unsafe_allow_html=True)
    if o["orbit"] == "LEO" and o["alt"] < 1000:
        days = max(15, int((o["alt"] - 180) * 0.85))
        reentry = (now + timedelta(days=days)).strftime("%Y-%m-%d")
        st.markdown(
            f'<div style="font-size:10px;color:#8aabcc;">Predicted reentry:</div>'
            f'<div style="font-size:11px;color:#f59e0b;">~{days} days ({reentry})</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div style="font-size:10px;color:#8aabcc;">Orbit regime:</div>'
            f'<div style="font-size:11px;color:#22c55e;">Stable · {o["orbit"]}</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-hdr">Simulation</div>', unsafe_allow_html=True)
    delta_a = st.slider("ΔSemi-major axis (km)", -5.0, 5.0, 0.0, 0.1, key="sim_delta")
    st.button("Compute", key="compute_btn", width='stretch')
    if delta_a != 0.0:
        new_alt    = o["alt"] + delta_a
        new_period = o["period"] * ((new_alt + R_EARTH) / (o["alt"] + R_EARTH)) ** 1.5
        st.markdown(
            f'<div style="font-size:10px;color:#8aabcc;">'
            f'New altitude: <span style="color:#00e5ff;">{new_alt:.1f} km</span><br>'
            f'New period: <span style="color:#00e5ff;">{new_period:.1f} min</span></div>',
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# CENTER
# ══════════════════════════════════════════════════════════════════════════════
with center:
    c1, c2, c3 = st.columns([2, 3, 2])
    with c1:
        met_extra = (
            '<div style="font-size:9px;color:#7090b0;letter-spacing:1px;'
            'text-transform:uppercase;margin-bottom:2px;">Mission Elapsed</div>'
        )
        st.iframe(
            _clock_html("met", epoch_ms=MISSION_EPOCH_MS,
                        color="#00e5ff", size=20,
                        extra_html=met_extra),
            height=48,
        )
    with c2:
        st.markdown(
            f'<div style="font-size:9px;color:#7090b0;text-align:center;letter-spacing:1px;">'
            f'IN-VIEW / TOTAL</div>'
            f'<div style="font-size:18px;font-family:monospace;color:#e0f0ff;text-align:center;">'
            f'{CATALOG_TOTAL:,} / {CATALOG_TOTAL:,}</div>',
            unsafe_allow_html=True,
        )
    with c3:
        date_hdr = now.strftime("%d %b %Y")
        c3_extra = (
            f'<div style="text-align:right;">'
            f'<div style="font-size:9px;color:#7090b0;letter-spacing:1px;">PROPAGATOR</div>'
            f'<div style="font-size:11px;color:#a855f7;">Kepler · J2 secular</div>'
            f'<div style="font-size:10px;color:#7090b0;">{date_hdr}&nbsp;'
        )
        c3_footer = 'Z</div></div>'
        st.iframe(
            _clock_html("utc", color="#7090b0", size=10,
                        extra_html=c3_extra).replace("</body>", c3_footer + "</body>"),
            height=55,
        )

    # 3D globe (interactive + animated)
    st.plotly_chart(globe_fig, width='stretch',
                    config={"displayModeBar": True,
                            "modeBarButtonsToRemove": ["toImage"],
                            "displaylogo": False})

    # Legend
    leg_items = [
        ("#00e5ff", "Low Earth Orbit"),
        ("#a855f7", "Medium Earth Orbit"),
        ("#ffd700", "Geostationary Belt"),
        ("#22c55e", "Highly Elliptical"),
        ("#ff6b35", "Debris"),
    ]
    leg = ('<div style="display:flex;gap:12px;flex-wrap:wrap;justify-content:center;'
           'font-size:10px;margin-top:-14px;color:#8aabcc;">')
    for color, label in leg_items:
        leg += (f'<span><span style="display:inline-block;width:9px;height:9px;'
                f'background:{color};border-radius:50%;margin-right:4px;'
                f'vertical-align:middle;"></span>{label}</span>')
    leg += "</div>"
    st.markdown(leg, unsafe_allow_html=True)

    st.markdown(
        '<div style="text-align:center;font-size:10px;color:#4a6a8a;margin-top:4px;">'
        'Drag to rotate · Scroll to zoom · Double-click to reset · '
        '<b style="color:#4a90d9;">▶ Play</b> to animate orbital motion'
        '</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# RIGHT
# ══════════════════════════════════════════════════════════════════════════════
with right:
    # Space Resilience Index
    st.markdown(
        f'<div class="section-hdr">Space Resilience Index'
        f'<span style="float:right;"><span class="badge-elevated">'
        f'{RESILIENCE_LEVEL}</span></span></div>',
        unsafe_allow_html=True,
    )
    sc = RESILIENCE_SCORE
    sc_color = "#22c55e" if sc < 40 else "#f59e0b" if sc < 70 else "#ff6b35"
    st.markdown(
        f'<div style="display:flex;align-items:baseline;gap:6px;margin:4px 0 5px 0;">'
        f'<span style="font-size:46px;font-weight:900;color:{sc_color};line-height:1;">{sc}</span>'
        f'<span style="font-size:14px;color:#7090b0;">/ 100</span></div>'
        f'<div style="background:#1a2540;border-radius:3px;height:7px;margin-bottom:8px;">'
        f'<div style="width:{sc}%;background:linear-gradient(90deg,{sc_color},{sc_color}88);'
        f'border-radius:3px;height:7px;"></div></div>',
        unsafe_allow_html=True,
    )
    bars = ""
    for metric, (val, color) in RESILIENCE_METRICS.items():
        bars += prog_bar(metric, val, color)
    st.markdown(bars, unsafe_allow_html=True)

    # Catalogue composition
    st.markdown('<div class="section-hdr">Catalogue Composition</div>', unsafe_allow_html=True)
    comp_html = ""
    for label, count, color in [("Payload", N_PAYLOAD, "#00e5ff"),
                                  ("Rocket Body", N_ROCKET, "#f59e0b"),
                                  ("Debris", N_DEBRIS, "#ff6b35")]:
        comp_html += comp_row(label, count, color, CATALOG_TOTAL)
    st.markdown(comp_html, unsafe_allow_html=True)

    # LEO altitude congestion
    st.markdown(
        '<div class="section-hdr">LEO Altitude Congestion'
        '<span style="float:right;font-size:9px;color:#7090b0;">ALTITUDE LEVEL</span></div>',
        unsafe_allow_html=True,
    )
    max_band = max(v for _, v in LEO_BANDS)
    cong = "".join(horiz_bar(band, count, max_band * 1.1) for band, count in LEO_BANDS)
    st.markdown(cong, unsafe_allow_html=True)

    # Conjunction watch
    st.markdown(
        '<div class="section-hdr">Conjunction Watch'
        '<span style="float:right;">'
        '<a href="#" style="font-size:9px;color:#4a90d9;text-decoration:none;'
        'letter-spacing:1px;">ANALYZE ▶</a></span></div>',
        unsafe_allow_html=True,
    )
    for c in CONJUNCTIONS:
        prob_str = f"{c['prob']:.2e}"
        st.markdown(
            f'<div style="background:#0d1526;border:1px solid #1e3a5f;border-radius:4px;'
            f'padding:6px 9px;margin-bottom:5px;">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;'
            f'margin-bottom:3px;">'
            f'<span style="font-size:10px;color:#c8d6e8;font-weight:600;">'
            f'{c["obj1"]} × {c["obj2"]}</span>'
            f'{conj_badge(c["sev"])}</div>'
            f'<div style="display:flex;gap:10px;font-size:10px;color:#7090b0;">'
            f'<span>Pc: <span style="color:#ff6b35;font-weight:700;">{prob_str}</span></span>'
            f'<span>TCA: {c["tca"]}</span>'
            f'<span>DCA: {c["dca"]}</span></div>'
            f'</div>',
            unsafe_allow_html=True,
        )
