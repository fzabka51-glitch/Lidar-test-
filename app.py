import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.ndimage import laplace, gaussian_filter
from scipy.ndimage import maximum_filter, minimum_filter
import datashader as ds
import io

# --- SEITENKONFIGURATION ---
st.set_page_config(page_title="LiDAR Archäologie Pro", layout="wide")
st.title("🏛️ LiDAR Analyse & Anomalie-Detektion (Vollversion)")

# --- ARCHÄOLOGISCHE ANALYSE-FUNKTIONEN ---

def rasterize_points(df, res):
    """Blitzschnelle Rasterisierung von Millionen Punkten mittels Datashader."""
    cvs = ds.Canvas(
        plot_width=int((df.x.max() - df.x.min()) / res),
        plot_height=int((df.y.max() - df.y.min()) / res),
        x_range=(df.x.min(), df.x.max()),
        y_range=(df.y.min(), df.y.max())
    )
    agg = cvs.points(df, 'x', 'y', ds.mean('z'))
    return np.array(agg.values, dtype=np.float32)

def calculate_hillshade(data, azimuth=315, angle_altitude=45, res=1.0):
    """Berechnet ein Schummerungsbild (Hillshade)."""
    azimuth_rad = np.deg2rad(azimuth)
    altitude_rad = np.deg2rad(angle_altitude)
    gy, gx = np.gradient(data, res, res)
    slope = np.arctan(np.sqrt(gx**2 + gy**2))
    aspect = np.arctan2(-gy, gx)
    shade = (np.cos(altitude_rad) * np.cos(slope)) + \
            (np.sin(altitude_rad) * np.sin(slope) * np.cos(azimuth_rad - aspect))
    return ((shade + 1) / 2).astype(np.float32)

def calculate_multi_hillshade(data, res=1.0):
    """Multi-Directional Shading (MDS) aus 4 Richtungen."""
    h1 = calculate_hillshade(data, 315, 45, res)
    h2 = calculate_hillshade(data, 45, 45, res)
    h3 = calculate_hillshade(data, 135, 45, res)
    h4 = calculate_hillshade(data, 225, 45, res)
    return (h1 + h2 + h3 + h4) / 4.0

def calculate_lrm(data, sigma=15):
    """Local Relief Model (LRM) / Residual Topography."""
    smoothed = gaussian_filter(data, sigma=sigma)
    return data - smoothed

def calculate_slope(data, res=1.0):
    """Berechnet die Hangneigung in Grad."""
    gy, gx = np.gradient(data, res, res)
    slope_deg = np.rad2deg(np.arctan(np.sqrt(gx**2 + gy**2)))
    p_high = np.nanpercentile(slope_deg, 98)
    return np.clip(slope_deg, 0, p_high)

def calculate_curvature(data):
    """Berechnet die lokale Krümmung (Laplace)."""
    curv = -laplace(data)
    p_low, p_high = np.percentile(curv, (2, 98))
    curv_clipped = np.clip(curv, p_low, p_high)
    c_min, c_max = curv_clipped.min(), curv_clipped.max()
    if c_max > c_min:
        return (curv_clipped - c_min) / (c_max - c_min)
    return np.full_like(curv, 0.5)

def detect_anomalies(lrm_data, threshold_m=0.3, neighborhood=10):
    """Findet lokale Maxima (Hügel) und Minima (Erdställe/Pingen)."""
    local_max = maximum_filter(lrm_data, size=neighborhood) == lrm_data
    pos_anomalies = (lrm_data > threshold_m) & local_max
    local_min = minimum_filter(lrm_data, size=neighborhood) == lrm_data
    neg_anomalies = (lrm_data < -threshold_m) & local_min
    return np.where(pos_anomalies), np.where(neg_anomalies)

# --- SIDEBAR (STEUERUNG) ---
with st.sidebar:
    st.header("⚙️ Parameter")
    uploaded_file = st.file_uploader("XYZ Datei laden (.xyz, .txt)", type=["xyz", "txt"])
    
    st.subheader("Raster & Filter")
    grid_res = st.number_input("Auflösung (m)", 0.1, 10.0, 0.5, help="0.5m ist ideal für Erdställe")
    lrm_sigma = st.slider("LRM Glättung (Sigma)", 1, 50, 15)
    
    st.subheader("🔎 Anomalie-Suche")
    detect_on = st.toggle("Hügel & Erdställe markieren", value=False)
    sens_m = st.slider("Empfindlichkeit (Meter)", 0.05, 2.0, 0.3)
    
    st.subheader("3D-Eigenschaften")
    z_exag = st.slider("Z-Überhöhung", 0.1, 10.0, 2.0, step=0.1)
    
    st.subheader("Anzeige")
    view_mode = st.radio("Ansicht 2D:", ["Einzelansicht", "Gitter-Übersicht"])

# --- HAUPTBEREICH ---
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file, sep=r'\s+', header=None, names=['x','y','z'], dtype=np.float32)
        if len(df) > 2000000:
            df = df.sample(2000000, random_state=42)
            st.warning("⚠️ Datensatz auf 2 Mio. Punkte reduziert.")

        with st.spinner("Berechne alle Modelle..."):
            gz = rasterize_points(df, grid_res)
            gz = np.nan_to_num(gz, nan=np.nanmean(gz))
            
            lrm_raw = calculate_lrm(gz, lrm_sigma)
            
            # Anomalie Detektion
            pos_idx, neg_idx = (None, None)
            if detect_on:
                pos_idx, neg_idx = detect_anomalies(lrm_raw, threshold_m=sens_m)

            # LRM Normalisierung für Visualisierung
            p_low, p_high = np.percentile(lrm_raw, (5, 95))
            lrm_vis = np.clip(lrm_raw, p_low, p_high)
            lrm_vis = (lrm_vis - lrm_vis.min()) / (lrm_vis.max() - lrm_vis.min())

            # Weitere Modelle berechnen
            mds = calculate_multi_hillshade(gz, grid_res)
            slope = calculate_slope(gz, grid_res)
            curv = calculate_curvature(gz)
            # Composite Fusion
            comp = np.clip(mds + (lrm_vis - 0.5) * 0.4, 0, 1)
            
            analysis_models = {
                "Final Composite (Fusion)": (comp, "gray", False),
                "Restrelief (LRM)": (lrm_vis, "RdBu", True),
                "MDS Composite": (mds, "gray", False),
                "Hangneigung (Slope)": (slope, "plasma", True),
                "Krümmung (Curvature)": (curv, "RdYlGn", True)
            }

        tab1, tab2 = st.tabs(["🖼️ 2D-Analyse & Detektion", "🌐 3D-Prospektion"])

        with tab1:
            if view_mode == "Gitter-Übersicht":
                cols = st.columns(2)
                for i, (name, (data, cmap, _)) in enumerate(analysis_models.items()):
                    with cols[i % 2]:
                        fig, ax = plt.subplots()
                        ax.imshow(data, cmap=cmap, interpolation='none', origin='lower')
                        ax.set_title(name)
                        ax.axis('off')
                        st.pyplot(fig)
                        plt.close(fig)
            else:
                sel_2d = st.selectbox("Modell wählen:", list(analysis_models.keys()))
                data, cmap, _ = analysis_models[sel_2d]
                
                fig, ax = plt.subplots(figsize=(10, 8))
                ax.imshow(data, cmap=cmap, interpolation='none', origin='lower')
                
                if detect_on:
                    if pos_idx and len(pos_idx[0]) > 0:
                        ax.scatter(pos_idx[1], pos_idx[0], edgecolors='yellow', facecolors='none', s=100, label='Hügel-Verdacht')
                    if neg_idx and len(neg_idx[0]) > 0:
                        ax.scatter(neg_idx[1], neg_idx[0], edgecolors='cyan', facecolors='none', s=100, label='Erdstall/Pinge')
                    ax.legend(loc='upper right')
                
                ax.axis('off')
                st.pyplot(fig)
                if detect_on:
                    st.write(f"Gefundene Hügel: {len(pos_idx[0]) if pos_idx else 0} | Gefundene Senken: {len(neg_idx[0]) if neg_idx else 0}")

        with tab2:
            selected_texture = st.selectbox("Textur für 3D wählen:", list(analysis_models.keys()), index=1)
            tex_data, tex_cmap, show_scale = analysis_models[selected_texture]
            
            step = max(1, int(np.sqrt(gz.size / 400000)))
            z_plot = gz[::step, ::step]
            surface_tex = tex_data[::step, ::step]

            fig3d = go.Figure(data=[go.Surface(
                z=z_plot, 
                surfacecolor=surface_tex, 
                colorscale=tex_cmap,
                showscale=show_scale,
                lighting=dict(ambient=0.6, diffuse=0.8)
            )])
            
            fig3d.update_layout(
                scene=dict(
                    aspectmode='data',
                    aspectratio=dict(x=1, y=1, z=z_exag),
                    xaxis=dict(visible=False), yaxis=dict(visible=False),
                    zaxis=dict(title="Höhe (m)")
                ),
                height=850, margin=dict(l=0, r=0, b=0, t=40)
            )
            st.plotly_chart(fig3d, use_container_width=True)

    except Exception as e:
        st.error(f"Fehler: {e}")
else:
    st.info("Bitte XYZ-Datei hochladen.")
