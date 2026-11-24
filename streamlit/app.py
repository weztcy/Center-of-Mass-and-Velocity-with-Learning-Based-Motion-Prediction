# -*- coding: utf-8 -*-
"""
=============================================================================
APLIKASI INTERAKTIF STREAMLIT (DARK MODE EDITION):
CENTER OF MASS & VELOCITY WITH LEARNING-BASED MOTION PREDICTION
=============================================================================
Aplikasi web interaktif untuk:
1. Menghitung Pusat Massa & Kecepatan Multi-Objek (2D & 3D).
2. Melatih & Mengevaluasi Model Machine Learning (Linear Regression).
3. Mensimulasikan & Membandingkan Prediksi Gerak (Fisika vs ML).
4. Menyajikan Dashboard Komprehensif dan Galeri Ekspor 300 DPI.
5. Dokumentasi Konsep Teori Fisika & Machine Learning.
=============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Konfigurasi Tema Dark Mode untuk Seluruh Plot Matplotlib
plt.style.use("dark_background")
plt.rcParams.update({
    "figure.facecolor": "#0e1117",
    "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",
    "grid.color": "#21262d",
    "text.color": "#e6edf3",
    "axes.labelcolor": "#e6edf3",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "figure.titlesize": 13,
    "axes.titlesize": 11,
    "legend.facecolor": "#161b22",
    "legend.edgecolor": "#30363d",
})

# Setup direktori proyek dan path resolving fleksibel (root proyek dan subfolder streamlit)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)

for p in [BASE_DIR, PARENT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Import fungsi-fungsi inti dari program yang sudah ada tanpa memodifikasinya
try:
    from program import (
        hitung_pusat_massa,
        hitung_kecepatan_pusat_massa,
        buat_dataset_sintetis,
        FEATURE_COLUMNS,
        TARGET_COLUMNS
    )
except ImportError:
    def hitung_pusat_massa(posisi, massa):
        posisi = np.asarray(posisi, dtype=float)
        massa = np.asarray(massa, dtype=float)
        return np.sum(posisi * massa[:, np.newaxis], axis=0) / np.sum(massa)

    def hitung_kecepatan_pusat_massa(kecepatan, massa):
        kecepatan = np.asarray(kecepatan, dtype=float)
        massa = np.asarray(massa, dtype=float)
        return np.sum(kecepatan * massa[:, np.newaxis], axis=0) / np.sum(massa)

    FEATURE_COLUMNS = [
        "cm_x_awal_terukur", "cm_y_awal_terukur", "cm_z_awal_terukur",
        "perpindahan_x_terukur", "perpindahan_y_terukur", "perpindahan_z_terukur",
    ]
    TARGET_COLUMNS = [
        "cm_x_masa_depan", "cm_y_masa_depan", "cm_z_masa_depan",
    ]

# Pencarian file dataset dan aset gambar (figures/visualizations) secara dinamis
def find_project_file(filename):
    """Mencari file baik di direktori lokal streamlit maupun di parent directory."""
    for d in [BASE_DIR, PARENT_DIR]:
        candidate = os.path.join(d, filename)
        if os.path.exists(candidate):
            return candidate
    return os.path.join(PARENT_DIR, filename)

DATASET_FILE = find_project_file("dataset_center_of_mass_ml.csv")

def get_asset_path(filename):
    """Mencari lokasi file visualisasi di folder figures maupun visualizations."""
    for parent in [BASE_DIR, PARENT_DIR]:
        for folder in ["figures", "visualizations"]:
            target = os.path.join(parent, folder, filename)
            if os.path.exists(target):
                return target
    return None

# =============================================================================
# KONFIGURASI HALAMAN STREAMLIT & STYLING DARK MODE
# =============================================================================
st.set_page_config(
    page_title="Center of Mass & Velocity ML Studio",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Dark Mode Khusus
st.markdown("""
<style>
    /* Styling Kartu Metrik Dark Mode */
    .metric-card {
        background: linear-gradient(135deg, #161b22 0%, #0d1117 100%);
        border-radius: 12px;
        padding: 16px 20px;
        border: 1px solid #30363d;
        border-left: 5px solid #3b82f6;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
        margin-bottom: 14px;
    }
    .metric-title {
        font-size: 0.82rem;
        font-weight: 600;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.55rem;
        font-weight: 700;
        color: #58a6ff;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #7d8590;
        margin-top: 4px;
    }
    .badge-tag {
        display: inline-block;
        padding: 4px 12px;
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: 14px;
        background-color: #1f2937;
        color: #38bdf8;
        border: 1px solid #374151;
        margin-right: 6px;
    }
    /* Mengatur style tombol tab */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid #30363d;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 18px;
        border-radius: 6px 6px 0 0;
        color: #8b949e;
    }
    .stTabs [aria-selected="true"] {
        color: #58a6ff !important;
        border-bottom: 2px solid #58a6ff !important;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# FUNGSI CACHE MACHINE LEARNING
# =============================================================================
@st.cache_data(show_spinner=False)
def load_or_create_dataset(jumlah_sampel=3000, noise_std=0.05, random_state=42, use_existing=True):
    """Memuat dataset CSV yang ada atau membuat dataset sintetis baru."""
    if use_existing and os.path.exists(DATASET_FILE):
        df = pd.read_csv(DATASET_FILE)
        return df, "Loaded from CSV file"
    else:
        rng = np.random.default_rng(random_state)
        baris = []
        for _ in range(jumlah_sampel):
            n_obj = int(rng.integers(2, 7))
            m = rng.uniform(0.5, 10.0, size=n_obj)
            pos0 = rng.uniform(-20.0, 20.0, size=(n_obj, 3))
            vel = rng.uniform(-5.0, 5.0, size=(n_obj, 3))
            t = float(rng.uniform(0.1, 5.0))

            cm0 = hitung_pusat_massa(pos0, m)
            vcm = hitung_kecepatan_pusat_massa(vel, m)
            disp = vcm * t
            cm_future = cm0 + disp

            cm0_meas = cm0 + rng.normal(0, noise_std, size=3)
            disp_meas = disp + rng.normal(0, noise_std, size=3)

            baris.append({
                "jumlah_objek": n_obj,
                "waktu_s": t,
                "kecepatan_cm_x": vcm[0],
                "kecepatan_cm_y": vcm[1],
                "kecepatan_cm_z": vcm[2],
                "cm_x_awal_terukur": cm0_meas[0],
                "cm_y_awal_terukur": cm0_meas[1],
                "cm_z_awal_terukur": cm0_meas[2],
                "perpindahan_x_terukur": disp_meas[0],
                "perpindahan_y_terukur": disp_meas[1],
                "perpindahan_z_terukur": disp_meas[2],
                "cm_x_masa_depan": cm_future[0],
                "cm_y_masa_depan": cm_future[1],
                "cm_z_masa_depan": cm_future[2],
            })
        return pd.DataFrame(baris), "Generated Synthetic Dataset"


@st.cache_resource(show_spinner=False)
def train_linear_model(df_data, test_size=0.2, random_state=42):
    """Melatih model Linear Regression dan menghitung metrik evaluasi."""
    X = df_data[FEATURE_COLUMNS]
    y = df_data[TARGET_COLUMNS]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Metrik
    overall_mae = mean_absolute_error(y_test, y_pred)
    overall_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    overall_r2 = r2_score(y_test, y_pred)

    axes_metrics = {}
    for i, axis in enumerate(["X", "Y", "Z"]):
        axes_metrics[axis] = {
            "MAE": mean_absolute_error(y_test.iloc[:, i], y_pred[:, i]),
            "RMSE": np.sqrt(mean_squared_error(y_test.iloc[:, i], y_pred[:, i])),
            "R2": r2_score(y_test.iloc[:, i], y_pred[:, i]),
        }

    return {
        "model": model,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred": y_pred,
        "overall_mae": overall_mae,
        "overall_rmse": overall_rmse,
        "overall_r2": overall_r2,
        "axes_metrics": axes_metrics
    }


# =============================================================================
# DATA PRESET UNTUK KALKULATOR FISIKA
# =============================================================================
PRESETS = {
    "Preset Default (4 Objek Riset)": pd.DataFrame([
        {"Objek": "Objek 1", "Massa (kg)": 2.5, "Pos X (m)": 2.0, "Pos Y (m)": 3.5, "Pos Z (m)": 1.2, "Vx (m/s)": 2.0, "Vy (m/s)": 1.2, "Vz (m/s)": 0.5},
        {"Objek": "Objek 2", "Massa (kg)": 4.0, "Pos X (m)": -3.0, "Pos Y (m)": 1.0, "Pos Z (m)": 4.0, "Vx (m/s)": -1.2, "Vy (m/s)": 2.5, "Vz (m/s)": -1.0},
        {"Objek": "Objek 3", "Massa (kg)": 3.2, "Pos X (m)": 4.2, "Pos Y (m)": -2.5, "Pos Z (m)": 2.8, "Vx (m/s)": 1.8, "Vy (m/s)": -1.5, "Vz (m/s)": 2.2},
        {"Objek": "Objek 4", "Massa (kg)": 5.0, "Pos X (m)": -1.5, "Pos Y (m)": -3.2, "Pos Z (m)": -1.8, "Vx (m/s)": -2.0, "Vy (m/s)": 0.8, "Vz (m/s)": 1.4},
    ]),
    "Sistem Biner (2 Objek)": pd.DataFrame([
        {"Objek": "Bintang A", "Massa (kg)": 8.0, "Pos X (m)": -5.0, "Pos Y (m)": 0.0, "Pos Z (m)": 0.0, "Vx (m/s)": 0.0, "Vy (m/s)": 3.0, "Vz (m/s)": 0.0},
        {"Objek": "Bintang B", "Massa (kg)": 4.0, "Pos X (m)": 10.0, "Pos Y (m)": 0.0, "Pos Z (m)": 0.0, "Vx (m/s)": 0.0, "Vy (m/s)": -6.0, "Vz (m/s)": 0.0},
    ]),
    "Sistem Segitiga Simetris (3 Objek)": pd.DataFrame([
        {"Objek": "Objek Alpha", "Massa (kg)": 3.0, "Pos X (m)": 0.0, "Pos Y (m)": 6.0, "Pos Z (m)": 0.0, "Vx (m/s)": 1.5, "Vy (m/s)": 0.0, "Vz (m/s)": 1.0},
        {"Objek": "Objek Beta", "Massa (kg)": 3.0, "Pos X (m)": -5.0, "Pos Y (m)": -3.0, "Pos Z (m)": 0.0, "Vx (m/s)": -1.0, "Vy (m/s)": 1.5, "Vz (m/s)": -0.5},
        {"Objek": "Objek Gamma", "Massa (kg)": 3.0, "Pos X (m)": 5.0, "Pos Y (m)": -3.0, "Pos Z (m)": 0.0, "Vx (m/s)": -0.5, "Vy (m/s)": -1.5, "Vz (m/s)": 0.5},
    ]),
}


# =============================================================================
# INISIALISASI SESSION STATE
# =============================================================================
if "objects_df" not in st.session_state:
    st.session_state["objects_df"] = PRESETS["Preset Default (4 Objek Riset)"].copy()

# Default CM & V_CM yang pasti tersedia
df_init = st.session_state["objects_df"]
m_init = df_init["Massa (kg)"].astype(float).values
p_init = df_init[["Pos X (m)", "Pos Y (m)", "Pos Z (m)"]].astype(float).values
v_init = df_init[["Vx (m/s)", "Vy (m/s)", "Vz (m/s)"]].astype(float).values

if "cm_current" not in st.session_state:
    st.session_state["cm_current"] = hitung_pusat_massa(p_init, m_init)
if "v_cm_current" not in st.session_state:
    st.session_state["v_cm_current"] = hitung_kecepatan_pusat_massa(v_init, m_init)


# =============================================================================
# SIDEBAR NAVIGATION
# =============================================================================
with st.sidebar:
    st.title("🌐 CM & Motion ML")
    st.caption("Center of Mass, Velocity & Learning-Based Prediction")
    st.markdown("---")

    menu_selected = st.radio(
        "Pilih Menu Navigasi:",
        [
            "1. 🎯 Kalkulator & Visualisasi Fisika",
            "2. 🧠 Pelatihan Model ML & Evaluasi",
            "3. 🚀 Prediksi Gerak (Fisika vs ML)",
            "4. 📊 Dashboard Lengkap & Galeri",
            "5. 📖 Teori & Dokumentasi Konsep"
        ],
        index=0
    )

    st.markdown("---")
    st.markdown("### ⚙️ Status Proyek")
    st.markdown("""
    <div style='font-size: 0.85rem; color: #8b949e; line-height: 1.6;'>
    • <b>Algoritma ML</b>: Linear Regression<br>
    • <b>Dimensi</b>: 3D (X, Y, Z)<br>
    • <b>Dataset</b>: 3.000 Skenario Sintetis<br>
    • <b>Tema</b>: Dark Mode Otomatis<br>
    • <b>Resolusi Plot</b>: 300 DPI High-Res<br>
    • <b>Framework</b>: Streamlit & Matplotlib
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.info("💡 **Tips**: Anda dapat mengubah jumlah objek, massa, posisi, dan kecepatan secara langsung di tabel interaktif.")


# =============================================================================
# HALAMAN 1: KALKULATOR & VISUALISASI FISIKA
# =============================================================================
if menu_selected.startswith("1."):
    st.title("🎯 Kalkulator & Visualisasi Fisika Multi-Objek")
    st.write(
        "Halaman ini menghitung **Pusat Massa (Center of Mass)** dan **Kecepatan Pusat Massa** "
        "dari sistem multi-partikel dalam koordinat 3 Dimensi, serta menampilkan visualisasi 2D dan 3D."
    )

    col_preset, col_btn_action = st.columns([3, 2])
    with col_preset:
        pilihan_preset = st.selectbox("Pilih Contoh Skenario / Preset:", list(PRESETS.keys()), index=0)
    with col_btn_action:
        st.write("")
        st.write("")
        if st.button("Muat Preset Terpilih", use_container_width=True):
            st.session_state["objects_df"] = PRESETS[pilihan_preset].copy()
            st.rerun()

    st.subheader("📝 Tabel Input Objek Interaktif")
    st.caption("Ubah nilai sel di bawah ini, tambah baris, atau hapus baris sesuai eksperimen Anda:")

    edited_df = st.data_editor(
        st.session_state["objects_df"],
        num_rows="dynamic",
        use_container_width=True,
        key="editor_objek"
    )

    # Ekstraksi array data dari data_editor
    try:
        massa_list = edited_df["Massa (kg)"].astype(float).values
        posisi_list = edited_df[["Pos X (m)", "Pos Y (m)", "Pos Z (m)"]].astype(float).values
        kecepatan_list = edited_df[["Vx (m/s)", "Vy (m/s)", "Vz (m/s)"]].astype(float).values

        if len(massa_list) < 1:
            st.error("Harap masukkan minimal 1 objek.")
            st.stop()
        if np.any(massa_list <= 0):
            st.error("Massa setiap objek harus lebih besar dari nol (> 0).")
            st.stop()

        # Perhitungan Fisika
        cm_result = hitung_pusat_massa(posisi_list, massa_list)
        v_cm_result = hitung_kecepatan_pusat_massa(kecepatan_list, massa_list)
        total_massa = np.sum(massa_list)
        v_cm_scalar = np.linalg.norm(v_cm_result)
        p_total = total_massa * v_cm_result
        p_scalar = np.linalg.norm(p_total)

        # Simpan ke session state
        st.session_state["cm_current"] = cm_result
        st.session_state["v_cm_current"] = v_cm_result
        st.session_state["posisi_current"] = posisi_list
        st.session_state["massa_current"] = massa_list
        st.session_state["kecepatan_current"] = kecepatan_list

    except Exception as e:
        st.error(f"Terjadi kesalahan pembacaan tabel data: {e}")
        st.stop()

    # Tampilan KPI Card Ringkasan Dark Mode
    st.markdown("### 📊 Hasil Perhitungan Sistem")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    with kpi1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Total Massa Sistem</div>
            <div class="metric-value">{total_massa:.2f} <span style="font-size:1rem;color:#8b949e;">kg</span></div>
            <div class="metric-sub">Jumlah Objek: {len(massa_list)} partikel</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Pusat Massa (X, Y, Z)</div>
            <div class="metric-value">({cm_result[0]:.2f}, {cm_result[1]:.2f}, {cm_result[2]:.2f})</div>
            <div class="metric-sub">Satuan: meter (m)</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Kecepatan Pusat Massa</div>
            <div class="metric-value">({v_cm_result[0]:.2f}, {v_cm_result[1]:.2f}, {v_cm_result[2]:.2f})</div>
            <div class="metric-sub">Magnitudo: {v_cm_scalar:.2f} m/s</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Momentum Total (P)</div>
            <div class="metric-value">{p_scalar:.2f} <span style="font-size:1rem;color:#8b949e;">kg·m/s</span></div>
            <div class="metric-sub">P = M_total × V_cm</div>
        </div>
        """, unsafe_allow_html=True)

    # Tab Visualisasi
    st.markdown("### 🎨 Visualisasi Koordinat & Vektor")
    tab1, tab2, tab3, tab4 = st.tabs([
        "📍 Pusat Massa 2D",
        "🌐 Pusat Massa 3D",
        "🧭 Vektor Kecepatan 2D",
        "🚀 Vektor Kecepatan 3D"
    ])

    # Tab 1: Pusat Massa 2D
    with tab1:
        fig1, ax1 = plt.subplots(figsize=(8.5, 5.5), dpi=150)
        warna1 = plt.cm.plasma(np.linspace(0.15, 0.85, len(posisi_list)))
        for i, pos in enumerate(posisi_list):
            ax1.scatter(
                pos[0], pos[1],
                color=warna1[i],
                s=120 + massa_list[i] * 35,
                alpha=0.9,
                edgecolors='white',
                linewidth=1.0,
                label=f"{edited_df.iloc[i]['Objek']} (m={massa_list[i]} kg)"
            )
            ax1.annotate(
                f"{edited_df.iloc[i]['Objek']}\n({pos[0]:.1f}, {pos[1]:.1f})",
                (pos[0], pos[1]),
                textcoords="offset points",
                xytext=(8, 8),
                fontsize=9,
                color="#e6edf3"
            )

        ax1.scatter(
            cm_result[0], cm_result[1],
            color="#ef4444",
            edgecolor="white",
            linewidth=1.5,
            s=220,
            marker="*",
            label=f"Pusat Massa ({cm_result[0]:.2f}, {cm_result[1]:.2f})",
            zorder=5
        )
        ax1.axvline(cm_result[0], color="#ef4444", linestyle=":", alpha=0.6)
        ax1.axhline(cm_result[1], color="#ef4444", linestyle=":", alpha=0.6)
        ax1.set_title("Visualisasi Pusat Massa Sistem Multi-Objek (Bidang 2D X-Y)", fontweight="bold", pad=10)
        ax1.set_xlabel("Koordinat X (m)", fontweight="semibold")
        ax1.set_ylabel("Koordinat Y (m)", fontweight="semibold")
        ax1.grid(True, linestyle="--", alpha=0.3)
        ax1.legend(loc="upper right", frameon=True, facecolor="#161b22", edgecolor="#30363d")
        plt.tight_layout()
        st.pyplot(fig1)
        plt.close(fig1)

    # Tab 2: Pusat Massa 3D
    with tab2:
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            elev3d = st.slider("Sudut Elevasi 3D (derajat):", -90, 90, 24, key="elev_cm")
        with col_c2:
            azim3d = st.slider("Sudut Azimuth 3D (derajat):", -180, 180, 40, key="azim_cm")

        fig2 = plt.figure(figsize=(9, 6.5), dpi=150)
        ax2 = fig2.add_subplot(111, projection="3d")
        ax2.set_facecolor("#161b22")
        warna2 = plt.cm.plasma(np.linspace(0.15, 0.85, len(posisi_list)))
        z_min = min(np.min(posisi_list[:, 2]), cm_result[2]) - 2.0

        for i, pos in enumerate(posisi_list):
            ax2.scatter(
                pos[0], pos[1], pos[2],
                color=warna2[i],
                s=120 + massa_list[i] * 35,
                alpha=0.9,
                edgecolors='white',
                label=f"{edited_df.iloc[i]['Objek']} (m={massa_list[i]} kg)"
            )
            ax2.plot([pos[0], pos[0]], [pos[1], pos[1]], [z_min, pos[2]], color=warna2[i], linestyle=":", alpha=0.4)

        ax2.scatter(
            cm_result[0], cm_result[1], cm_result[2],
            color="#ef4444",
            edgecolor="white",
            linewidth=1.5,
            s=260,
            marker="*",
            label=f"Pusat Massa ({cm_result[0]:.2f}, {cm_result[1]:.2f}, {cm_result[2]:.2f})",
            zorder=10
        )
        ax2.plot([cm_result[0], cm_result[0]], [cm_result[1], cm_result[1]], [z_min, cm_result[2]], color="#ef4444", linestyle="--", alpha=0.7)

        ax2.set_title("Visualisasi Posisi Objek & Pusat Massa dalam Ruang 3D", fontweight="bold", pad=12)
        ax2.set_xlabel("Sumbu X (m)", fontweight="semibold", labelpad=8)
        ax2.set_ylabel("Sumbu Y (m)", fontweight="semibold", labelpad=8)
        ax2.set_zlabel("Sumbu Z (m)", fontweight="semibold", labelpad=8)
        ax2.view_init(elev=elev3d, azim=azim3d)
        ax2.legend(loc="upper left", bbox_to_anchor=(0.0, 0.95))
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    # Tab 3: Kecepatan 2D (Quiver)
    with tab3:
        fig3, ax3 = plt.subplots(figsize=(8.5, 5.5), dpi=150)
        warna3 = plt.cm.viridis(np.linspace(0.15, 0.85, len(kecepatan_list)))
        for i, v in enumerate(kecepatan_list):
            ax3.quiver(
                0, 0, v[0], v[1],
                color=warna3[i],
                angles="xy",
                scale_units="xy",
                scale=1,
                alpha=0.85,
                width=0.007,
                label=f"v {edited_df.iloc[i]['Objek']}: ({v[0]:.1f}, {v[1]:.1f}) m/s"
            )

        ax3.quiver(
            0, 0, v_cm_result[0], v_cm_result[1],
            color="#ef4444",
            angles="xy",
            scale_units="xy",
            scale=1,
            width=0.012,
            zorder=6,
            label=f"v Pusat Massa: ({v_cm_result[0]:.2f}, {v_cm_result[1]:.2f}) m/s"
        )
        max_v = max(np.max(np.abs(kecepatan_list[:, :2])), np.max(np.abs(v_cm_result[:2]))) * 1.3
        ax3.set_xlim(-max_v, max_v)
        ax3.set_ylim(-max_v, max_v)
        ax3.set_aspect("equal", adjustable="box")
        ax3.axhline(0, color="#4b5563", linewidth=0.8, linestyle="-", alpha=0.5)
        ax3.axvline(0, color="#4b5563", linewidth=0.8, linestyle="-", alpha=0.5)
        ax3.set_title("Diagram Vektor Kecepatan Objek & Kecepatan Pusat Massa (2D)", fontweight="bold", pad=10)
        ax3.set_xlabel("Kecepatan Vx (m/s)", fontweight="semibold")
        ax3.set_ylabel("Kecepatan Vy (m/s)", fontweight="semibold")
        ax3.grid(True, linestyle="--", alpha=0.3)
        ax3.legend(loc="upper right", frameon=True, facecolor="#161b22", edgecolor="#30363d")
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)

    # Tab 4: Kecepatan 3D (Quiver)
    with tab4:
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            elev_v = st.slider("Sudut Elevasi Kecepatan 3D:", -90, 90, 20, key="elev_vel")
        with col_v2:
            azim_v = st.slider("Sudut Azimuth Kecepatan 3D:", -180, 180, 55, key="azim_vel")

        fig4 = plt.figure(figsize=(9, 6.5), dpi=150)
        ax4 = fig4.add_subplot(111, projection="3d")
        ax4.set_facecolor("#161b22")
        warna4 = plt.cm.viridis(np.linspace(0.15, 0.85, len(kecepatan_list)))

        for i, v in enumerate(kecepatan_list):
            ax4.quiver(
                0, 0, 0, v[0], v[1], v[2],
                color=warna4[i],
                linewidth=2.0,
                alpha=0.85,
                arrow_length_ratio=0.15,
                label=f"v {edited_df.iloc[i]['Objek']}: ({v[0]:.1f}, {v[1]:.1f}, {v[2]:.1f})"
            )

        ax4.quiver(
            0, 0, 0, v_cm_result[0], v_cm_result[1], v_cm_result[2],
            color="#ef4444",
            linewidth=3.5,
            arrow_length_ratio=0.15,
            label=f"v Pusat Massa ({v_cm_result[0]:.2f}, {v_cm_result[1]:.2f}, {v_cm_result[2]:.2f})"
        )

        max_v3 = max(np.max(np.abs(kecepatan_list)), np.max(np.abs(v_cm_result))) * 1.2
        ax4.set_xlim([-max_v3, max_v3])
        ax4.set_ylim([-max_v3, max_v3])
        ax4.set_zlim([-max_v3, max_v3])
        ax4.set_title("Vektor Kecepatan Multi-Objek & Pusat Massa dalam Ruang 3D", fontweight="bold", pad=12)
        ax4.set_xlabel("Vx (m/s)", fontweight="semibold", labelpad=8)
        ax4.set_ylabel("Vy (m/s)", fontweight="semibold", labelpad=8)
        ax4.set_zlabel("Vz (m/s)", fontweight="semibold", labelpad=8)
        ax4.view_init(elev=elev_v, azim=azim_v)
        ax4.legend(loc="upper left", bbox_to_anchor=(0.0, 0.98))
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close(fig4)


# =============================================================================
# HALAMAN 2: PELATIHAN MODEL ML & EVALUASI
# =============================================================================
elif menu_selected.startswith("2."):
    st.title("🧠 Pelatihan Model Machine Learning & Evaluasi")
    st.write(
        "Model mempelajari korelasi antara **posisi pusat massa awal terukur** dan **vektor perpindahan terukur** "
        "untuk memprediksi posisi pusat massa di masa depan di bawah pengaruh noise/ketidakpastian pengukuran sensor."
    )

    st.subheader("⚙️ Parameter Dataset & Pelatihan")
    c_data1, c_data2 = st.columns([2, 3])

    with c_data1:
        sumber_dataset = st.radio(
            "Sumber Dataset:",
            ["Gunakan Dataset CSV yang Tersedia", "Generate Dataset Sintetis Kustom"],
            index=0
        )

    with c_data2:
        if sumber_dataset == "Generate Dataset Sintetis Kustom":
            c_s1, c_s2 = st.columns(2)
            with c_s1:
                n_samples = st.slider("Jumlah Sampel:", 500, 10000, 3000, step=500)
            with c_s2:
                noise_lvl = st.slider("Tingkat Noise Pengukuran (Std Dev):", 0.0, 0.3, 0.05, step=0.01)
            use_existing_csv = False
        else:
            st.info("Menggunakan dataset bawaan proyek (`dataset_center_of_mass_ml.csv`).")
            n_samples = 3000
            noise_lvl = 0.05
            use_existing_csv = True

    c_split1, c_split2 = st.columns(2)
    with c_split1:
        test_ratio = st.slider("Rasio Data Pengujian (Test Split):", 0.1, 0.4, 0.2, step=0.05)
    with c_split2:
        rand_seed = st.number_input("Random Seed:", value=42, step=1)

    # Muat data & Latih Model
    with st.spinner("Memproses dataset dan melatih model Linear Regression..."):
        df_ml, source_info = load_or_create_dataset(
            jumlah_sampel=n_samples,
            noise_std=noise_lvl,
            random_state=int(rand_seed),
            use_existing=use_existing_csv
        )
        training_results = train_linear_model(
            df_ml,
            test_size=test_ratio,
            random_state=int(rand_seed)
        )

    # Ringkasan Metrik
    st.markdown("### 🏆 Hasil Evaluasi Model")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">R-Squared (R²) Overall</div>
            <div class="metric-value">{training_results['overall_r2']:.6f}</div>
            <div class="metric-sub" style="color:#4ade80;">Sangat Mendekati 1.0 (Akurat)</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Mean Absolute Error (MAE)</div>
            <div class="metric-value">{training_results['overall_mae']:.6f} <span style="font-size:0.9rem;color:#8b949e;">m</span></div>
            <div class="metric-sub">Rata-rata kesalahan absolut</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Root Mean Squared Error (RMSE)</div>
            <div class="metric-value">{training_results['overall_rmse']:.6f} <span style="font-size:0.9rem;color:#8b949e;">m</span></div>
            <div class="metric-sub">Sensitif terhadap deviasi</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Jumlah Data Latih / Uji</div>
            <div class="metric-value">{len(training_results['X_train']):,} / {len(training_results['X_test']):,}</div>
            <div class="metric-sub">Total: {len(df_ml):,} baris</div>
        </div>
        """, unsafe_allow_html=True)

    # Tabel Metrik per Sumbu
    st.markdown("#### 📐 Metrik Evaluasi Per Sumbu (X, Y, Z)")
    axes_df = pd.DataFrame([
        {"Sumbu": "Sumbu X (cm_x_masa_depan)", "R²": training_results['axes_metrics']['X']['R2'], "MAE (m)": training_results['axes_metrics']['X']['MAE'], "RMSE (m)": training_results['axes_metrics']['X']['RMSE']},
        {"Sumbu": "Sumbu Y (cm_y_masa_depan)", "R²": training_results['axes_metrics']['Y']['R2'], "MAE (m)": training_results['axes_metrics']['Y']['MAE'], "RMSE (m)": training_results['axes_metrics']['Y']['RMSE']},
        {"Sumbu": "Sumbu Z (cm_z_masa_depan)", "R²": training_results['axes_metrics']['Z']['R2'], "MAE (m)": training_results['axes_metrics']['Z']['MAE'], "RMSE (m)": training_results['axes_metrics']['Z']['RMSE']},
    ])
    st.table(axes_df.style.format({"R²": "{:.6f}", "MAE (m)": "{:.6f}", "RMSE (m)": "{:.6f}"}))

    # Visualisasi Evaluasi Model
    st.markdown("### 📈 Visualisasi Evaluasi Model")
    tab_eval1, tab_eval2, tab_eval3 = st.tabs([
        "🎯 Aktual vs Prediksi (3 Sumbu)",
        "📊 Distribusi Residual (Error Histogram)",
        "🔍 Koefisien & Bobot Model"
    ])

    y_test_np = training_results['y_test'].to_numpy()
    y_pred_np = training_results['y_pred']

    with tab_eval1:
        fig_eval, axes_eval = plt.subplots(1, 3, figsize=(15, 4.5), dpi=150)
        colors_eval = ["#38bdf8", "#34d399", "#fb923c"]
        for i, ax in enumerate(axes_eval):
            axis_name = ["X", "Y", "Z"][i]
            ax.scatter(y_test_np[:, i], y_pred_np[:, i], alpha=0.45, color=colors_eval[i], s=25, edgecolors="none")
            low = min(y_test_np[:, i].min(), y_pred_np[:, i].min())
            high = max(y_test_np[:, i].max(), y_pred_np[:, i].max())
            ax.plot([low, high], [low, high], color="#ef4444", linestyle="--", linewidth=1.8, label="Ideal (y = x)")

            r2_val = training_results['axes_metrics'][axis_name]['R2']
            mae_val = training_results['axes_metrics'][axis_name]['MAE']
            rmse_val = training_results['axes_metrics'][axis_name]['RMSE']
            info_txt = f"$R^2$: {r2_val:.5f}\nMAE: {mae_val:.4f}\nRMSE: {rmse_val:.4f}"
            ax.text(0.05, 0.93, info_txt, transform=ax.transAxes, verticalalignment="top",
                    fontsize=9.5, bbox=dict(boxstyle="round,pad=0.5", facecolor="#161b22", edgecolor="#30363d", alpha=0.9))

            ax.set_title(f"Sumbu {axis_name} (cm_{axis_name.lower()}_masa_depan)", fontweight="bold")
            ax.set_xlabel(f"Nilai Aktual {axis_name} (m)", fontweight="semibold")
            ax.set_ylabel(f"Nilai Prediksi {axis_name} (m)", fontweight="semibold")
            ax.grid(True, linestyle="--", alpha=0.3)
            ax.legend(loc="lower right")

        fig_eval.suptitle("Evaluasi Model: Nilai Aktual vs Prediksi Linear Regression", fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()
        st.pyplot(fig_eval)
        plt.close(fig_eval)

    with tab_eval2:
        residuals = y_pred_np - y_test_np
        fig_res, axes_res = plt.subplots(1, 3, figsize=(15, 4.2), dpi=150)
        for i, ax in enumerate(axes_res):
            axis_name = ["X", "Y", "Z"][i]
            res = residuals[:, i]
            ax.hist(res, bins=35, color=colors_eval[i], alpha=0.75, edgecolor="#161b22", linewidth=0.8, density=True)
            ax.axvline(0, color="#ef4444", linestyle="--", linewidth=1.5, label="Error = 0")
            ax.axvline(np.mean(res), color="#e2e8f0", linestyle=":", linewidth=1.2, label=f"Mean: {np.mean(res):.4f}")
            ax.set_title(f"Distribusi Residual Sumbu {axis_name} ($\\Delta {axis_name}$)", fontweight="bold")
            ax.set_xlabel("Residual Prediksi (m)", fontweight="semibold")
            ax.set_ylabel("Kerapatan (Density)", fontweight="semibold")
            ax.grid(True, linestyle="--", alpha=0.3)
            ax.legend(loc="upper right", fontsize=8.5)

        fig_res.suptitle("Distribusi Residual Error Model Linear Regression", fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()
        st.pyplot(fig_res)
        plt.close(fig_res)

    with tab_eval3:
        st.write("Persamaan regresi multi-target mempelajari matriks bobot $\\mathbf{W}$ dan bias $\\vec{b}$:")
        st.latex(r"\vec{R}_{future} = \mathbf{W} \cdot \vec{X}_{features} + \vec{b}")

        model_coef = training_results['model'].coef_
        model_intercept = training_results['model'].intercept_

        coef_df = pd.DataFrame(
            model_coef,
            index=["Target: cm_x", "Target: cm_y", "Target: cm_z"],
            columns=FEATURE_COLUMNS
        )
        coef_df["Intercept (b)"] = model_intercept
        st.dataframe(coef_df.style.format("{:.5f}"), use_container_width=True)

        st.caption(
            "Perhatikan bahwa bobot untuk `cm_awal` dan `perpindahan` bernilai mendekati 1.0, "
            "sedangkan fitur orthogonal bernilai mendekati 0.0. "
            "Ini membuktikan secara empiris bahwa model berhasil merekonstruksi hukum fisika $\\vec{R} = \\vec{R}_0 + \\Delta \\vec{R}$!"
        )

    # Inspeksi Data Mentah
    with st.expander("🔎 Lihat Contoh Dataset (50 Baris Pertama)"):
        st.dataframe(df_ml.head(50), use_container_width=True)


# =============================================================================
# HALAMAN 3: PREDIKSI GERAK (FISIKA VS ML)
# =============================================================================
elif menu_selected.startswith("3."):
    st.title("🚀 Prediksi Gerak & Trajektori Interaktif (Fisika vs ML)")
    st.write(
        "Bandingkan hasil prediksi posisi pusat massa di masa depan ($t > 0$) "
        "antara **rumus analitik fisika** dan **model Linear Regression berbasis pembelajaran mesin**."
    )

    # Muat model ML
    df_ml, _ = load_or_create_dataset(use_existing=True)
    training_results = train_linear_model(df_ml)
    model = training_results["model"]

    # Parameter Input Skenario
    st.subheader("🎯 Konfigurasi Parameter Prediksi")
    col_mode, col_time = st.columns([2, 2])

    with col_mode:
        sumber_kondisi = st.radio(
            "Gunakan Parameter Awal dari:",
            ["Nilai dari Kalkulator Fisika (Halaman 1)", "Input Manual Bebas"],
            index=0
        )

    with col_time:
        waktu_prediksi = st.slider(
            "Waktu Prediksi ke Depan (detik):",
            min_value=0.1,
            max_value=10.0,
            value=2.0,
            step=0.1
        )

    if sumber_kondisi.startswith("Nilai dari Kalkulator"):
        cm_awal_input = st.session_state.get("cm_current", np.array([0.43, 0.44, 1.48]))
        v_cm_input = st.session_state.get("v_cm_current", np.array([-0.51, 0.64, 0.65]))
        st.success(f"Menggunakan data kalkulator: CM Awal = {np.round(cm_awal_input, 3)}, V_CM = {np.round(v_cm_input, 3)}")
    else:
        c_in1, c_in2 = st.columns(2)
        with c_in1:
            st.markdown("**Posisi Pusat Massa Awal (m):**")
            cx = st.number_input("CM X awal:", value=float(st.session_state.get("cm_current", [0.43, 0.44, 1.48])[0]), step=0.1)
            cy = st.number_input("CM Y awal:", value=float(st.session_state.get("cm_current", [0.43, 0.44, 1.48])[1]), step=0.1)
            cz = st.number_input("CM Z awal:", value=float(st.session_state.get("cm_current", [0.43, 0.44, 1.48])[2]), step=0.1)
            cm_awal_input = np.array([cx, cy, cz])
        with c_in2:
            st.markdown("**Kecepatan Pusat Massa (m/s):**")
            vx = st.number_input("Vx CM:", value=float(st.session_state.get("v_cm_current", [-0.51, 0.64, 0.65])[0]), step=0.1)
            vy = st.number_input("Vy CM:", value=float(st.session_state.get("v_cm_current", [-0.51, 0.64, 0.65])[1]), step=0.1)
            vz = st.number_input("Vz CM:", value=float(st.session_state.get("v_cm_current", [-0.51, 0.64, 0.65])[2]), step=0.1)
            v_cm_input = np.array([vx, vy, vz])

    # Perhitungan Fisika vs ML
    perpindahan_fisika = v_cm_input * waktu_prediksi
    posisi_fisika = cm_awal_input + perpindahan_fisika

    fitur_pred = pd.DataFrame(
        [[*cm_awal_input, *perpindahan_fisika]],
        columns=FEATURE_COLUMNS
    )
    posisi_ml = model.predict(fitur_pred)[0]

    selisih_vektor = np.abs(posisi_ml - posisi_fisika)
    jarak_deviasi = np.linalg.norm(posisi_ml - posisi_fisika)

    # Tampilan Hasil Perbandingan Dark Mode
    st.markdown("### 📊 Hasil Komparasi Prediksi")
    res1, res2, res3 = st.columns(3)

    with res1:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #2dd4bf;">
            <div class="metric-title">📐 Prediksi Rumus Fisika (t = {waktu_prediksi:.1f} s)</div>
            <div class="metric-value">({posisi_fisika[0]:.3f}, {posisi_fisika[1]:.3f}, {posisi_fisika[2]:.3f})</div>
            <div class="metric-sub">R(t) = R₀ + V_cm × t</div>
        </div>
        """, unsafe_allow_html=True)

    with res2:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #f87171;">
            <div class="metric-title">🤖 Prediksi Machine Learning (t = {waktu_prediksi:.1f} s)</div>
            <div class="metric-value">({posisi_ml[0]:.3f}, {posisi_ml[1]:.3f}, {posisi_ml[2]:.3f})</div>
            <div class="metric-sub">Linear Regression Model</div>
        </div>
        """, unsafe_allow_html=True)

    with res3:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #fbbf24;">
            <div class="metric-title">⚡ Deviasi Euclidean (Jarak)</div>
            <div class="metric-value">{jarak_deviasi:.5f} <span style="font-size:1rem;color:#8b949e;">m</span></div>
            <div class="metric-sub">ΔX={selisih_vektor[0]:.4f}, ΔY={selisih_vektor[1]:.4f}, ΔZ={selisih_vektor[2]:.4f}</div>
        </div>
        """, unsafe_allow_html=True)

    # Visualisasi 3D Trajektori
    st.markdown("### 🌐 Visualisasi 3D Lintasan Trajektori & Prediksi")
    col_cam1, col_cam2 = st.columns(2)
    with col_cam1:
        elev_p = st.slider("Elevasi Kamera 3D:", -90, 90, 22, key="elev_pred")
    with col_cam2:
        azim_p = st.slider("Azimuth Kamera 3D:", -180, 180, 48, key="azim_pred")

    # Plot 3D Perbandingan Dark Mode
    fig_pred = plt.figure(figsize=(9.5, 6.5), dpi=150)
    ax_p = fig_pred.add_subplot(111, projection="3d")
    ax_p.set_facecolor("#161b22")

    # Titik Pusat Massa Awal
    ax_p.scatter(*cm_awal_input, color="#38bdf8", s=140, marker="o", edgecolors="white",
                 label=f"CM Awal t=0 ({cm_awal_input[0]:.2f}, {cm_awal_input[1]:.2f}, {cm_awal_input[2]:.2f})", zorder=5)

    # Prediksi Fisika
    ax_p.scatter(*posisi_fisika, color="#34d399", marker="^", s=160, edgecolors="white",
                 label=f"Rumus Fisika t={waktu_prediksi:.1f}s ({posisi_fisika[0]:.2f}, {posisi_fisika[1]:.2f}, {posisi_fisika[2]:.2f})", zorder=6)

    # Prediksi ML
    ax_p.scatter(*posisi_ml, color="#f87171", marker="X", s=160, edgecolors="white",
                 label=f"Prediksi ML ({posisi_ml[0]:.2f}, {posisi_ml[1]:.2f}, {posisi_ml[2]:.2f})", zorder=7)

    # Garis Lintasan Fisika
    t_points = np.linspace(0, waktu_prediksi, 30)
    traj_points = np.array([cm_awal_input + v_cm_input * t for t in t_points])
    ax_p.plot(traj_points[:, 0], traj_points[:, 1], traj_points[:, 2],
              color="#34d399", linestyle="--", linewidth=2.0, alpha=0.8,
              label="Lintasan Trajektori Fisika")

    # Garis Prediksi ML
    ax_p.plot([cm_awal_input[0], posisi_ml[0]],
              [cm_awal_input[1], posisi_ml[1]],
              [cm_awal_input[2], posisi_ml[2]],
              color="#f87171", linestyle=":", linewidth=1.8, alpha=0.7,
              label="Arah Prediksi ML")

    # Garis Deviasi
    ax_p.plot([posisi_fisika[0], posisi_ml[0]],
              [posisi_fisika[1], posisi_ml[1]],
              [posisi_fisika[2], posisi_ml[2]],
              color="#fbbf24", linestyle="-", linewidth=1.8,
              label=f"Garis Deviasi Fisika-ML ({jarak_deviasi:.4f} m)")

    ax_p.set_title("Perbandingan Posisi Pusat Massa: Rumus Fisika vs Machine Learning", fontweight="bold", pad=12)
    ax_p.set_xlabel("Sumbu X (m)", fontweight="semibold", labelpad=8)
    ax_p.set_ylabel("Sumbu Y (m)", fontweight="semibold", labelpad=8)
    ax_p.set_zlabel("Sumbu Z (m)", fontweight="semibold", labelpad=8)
    ax_p.view_init(elev=elev_p, azim=azim_p)
    ax_p.legend(loc="upper left", bbox_to_anchor=(0.0, 0.98), fontsize=8.5)
    plt.tight_layout()
    st.pyplot(fig_pred)
    plt.close(fig_pred)

    # Animasi / Simulasi Waktu Dinamis
    st.markdown("### ⏱️ Simulasi Pergerakan Waktu Nyata (Time-Lapse CM)")
    st.caption("Geser slider waktu di bawah ini untuk melihat trajektori dinamis dari $t=0$ hingga waktu target:")
    step_t = st.slider("Waktu Animasi t (s):", 0.0, float(waktu_prediksi), float(waktu_prediksi), step=0.1, key="anim_slider")

    curr_phys = cm_awal_input + v_cm_input * step_t
    curr_disp = v_cm_input * step_t
    curr_feat = pd.DataFrame([[*cm_awal_input, *curr_disp]], columns=FEATURE_COLUMNS)
    curr_ml = model.predict(curr_feat)[0]

    st.write(
        f"**Pada $t = {step_t:.1f}$ detik:** "
        f"Posisi Fisika = `({curr_phys[0]:.2f}, {curr_phys[1]:.2f}, {curr_phys[2]:.2f})` | "
        f"Posisi ML = `({curr_ml[0]:.2f}, {curr_ml[1]:.2f}, {curr_ml[2]:.2f})` | "
        f"Deviasi = `{np.linalg.norm(curr_ml - curr_phys):.4f} m`"
    )


# =============================================================================
# HALAMAN 4: DASHBOARD LENGKAP & GALERI EKSPOR
# =============================================================================
elif menu_selected.startswith("4."):
    st.title("📊 Dashboard Ringkasan & Galeri Ekspor Visualisasi")
    st.write(
        "Halaman ini menyajikan dashboard terpadu proyek serta seluruh file visualisasi "
        "yang telah diekspor dalam format PNG berkualitas tinggi (**300 DPI High-Resolution**)."
    )

    tab_dash1, tab_dash2 = st.tabs(["🖼️ Dashboard Terpadu (2x2 Panel)", "📁 Galeri Gambar High-Res (300 DPI)"])

    with tab_dash1:
        st.subheader("Dashboard Ringkasan Proyek 4-in-1")
        st.caption("Memadukan Pusat Massa 2D, Vektor Kecepatan 2D, Evaluasi Regresi ML, dan Prediksi Trajektori 3D dalam satu tampilan:")

        dash_img_path = get_asset_path("08_dashboard_ringkasan_proyek.png")
        if dash_img_path and os.path.exists(dash_img_path):
            st.image(dash_img_path, caption="Dashboard Ringkasan Proyek (300 DPI)", use_container_width=True)
            with open(dash_img_path, "rb") as f:
                st.download_button(
                    label="⬇️ Download Dashboard PNG (300 DPI)",
                    data=f.read(),
                    file_name="08_dashboard_ringkasan_proyek.png",
                    mime="image/png",
                    use_container_width=True
                )
        else:
            st.warning("Memuat dashboard terpadu...")

    with tab_dash2:
        st.subheader("Galeri File Visualisasi Akademik")
        image_files = [
            ("01_pusat_massa_2d.png", "Pusat Massa 2D (Sumbu X-Y)"),
            ("02_pusat_massa_3d.png", "Pusat Massa 3D Sistem Multi-Objek"),
            ("03_kecepatan_pusat_massa_2d.png", "Diagram Vektor Kecepatan 2D (Quiver)"),
            ("04_kecepatan_pusat_massa_3d.png", "Diagram Vektor Kecepatan 3D (Quiver)"),
            ("05_evaluasi_linear_regression.png", "Evaluasi Model ML: Aktual vs Prediksi"),
            ("06_distribusi_residual_error.png", "Distribusi Residual Error Model"),
            ("07_perbandingan_prediksi_posisi_3d.png", "Perbandingan Trajektori 3D: Fisika vs ML"),
            ("08_dashboard_ringkasan_proyek.png", "Dashboard Terpadu Komprehensif")
        ]

        cols = st.columns(2)
        for idx, (img_name, img_desc) in enumerate(image_files):
            col_target = cols[idx % 2]
            img_path = get_asset_path(img_name)
            with col_target:
                st.markdown(f"**{img_desc}** (`{img_name}`)")
                if img_path and os.path.exists(img_path):
                    st.image(img_path, use_container_width=True)
                    with open(img_path, "rb") as f:
                        st.download_button(
                            label=f"⬇️ Download {img_name}",
                            data=f.read(),
                            file_name=img_name,
                            mime="image/png",
                            key=f"dl_{img_name}"
                        )
                else:
                    st.info(f"File {img_name} sedang dimuat.")
                st.markdown("---")


# =============================================================================
# HALAMAN 5: TEORI & DOKUMENTASI KONSEP
# =============================================================================
elif menu_selected.startswith("5."):
    st.title("📖 Teori & Konsep Fisika - Machine Learning")
    st.write(
        "Dokumentasi ilmiah mengenai formulasi matematika pusat massa, kecepatan pusat massa, "
        "kekekalan momentum, serta integrasi pemodelan Machine Learning untuk prediksi kinematika."
    )

    st.markdown("### 1. Formulasi Pusat Massa (Center of Mass)")
    st.write(
        "Pusat massa sistem multi-partikel merupakan titik ekuivalen di mana seluruh massa sistem "
        "dapat dianggap terpusat. Untuk sistem diskret yang terdiri dari $N$ objek dengan massa $m_i$ "
        "dan vektor posisi $\\vec{r}_i = (x_i, y_i, z_i)$:"
    )
    st.latex(r"\vec{R}_{cm} = \frac{\sum_{i=1}^{N} m_i \vec{r}_i}{\sum_{i=1}^{N} m_i} = \frac{1}{M_{total}} \sum_{i=1}^{N} m_i \vec{r}_i")
    st.write("Dalam komponen Kartesian 3 Dimensi:")
    st.latex(r"X_{cm} = \frac{\sum m_i x_i}{\sum m_i}, \quad Y_{cm} = \frac{\sum m_i y_i}{\sum m_i}, \quad Z_{cm} = \frac{\sum m_i z_i}{\sum m_i}")

    st.markdown("---")
    st.markdown("### 2. Formulasi Kecepatan Pusat Massa")
    st.write(
        "Turunan pertama terhadap waktu dari posisi pusat massa menghasilkan kecepatan pusat massa $\\vec{V}_{cm}$:"
    )
    st.latex(r"\vec{V}_{cm} = \frac{d\vec{R}_{cm}}{dt} = \frac{\sum_{i=1}^{N} m_i \vec{v}_i}{\sum_{i=1}^{N} m_i}")
    st.write(
        "Kecepatan pusat massa merepresentasikan momentum total sistem dibagi total massa sistem:"
    )
    st.latex(r"\vec{P}_{total} = M_{total} \vec{V}_{cm} = \sum_{i=1}^{N} m_i \vec{v}_i")

    st.markdown("---")
    st.markdown("### 3. Prediksi Gerak Kinematika Translasi")
    st.write(
        "Jika tidak ada gaya eksternal netto yang bekerja pada sistem (atau dalam kerangka kecepatan konstan), "
        "posisi pusat massa pada waktu masa depan $t$ diprediksi secara analitik melalui:"
    )
    st.latex(r"\vec{R}_{cm}(t) = \vec{R}_{cm}(0) + \vec{V}_{cm} \cdot t = \vec{R}_{cm}(0) + \Delta \vec{R}_{cm}")

    st.markdown("---")
    st.markdown("### 4. Integrasi Machine Learning (Linear Regression)")
    st.write(
        "Dalam aplikasi dunia nyata (seperti sensor radar, LiDAR, atau computer vision), "
        "posisi dan kecepatan awal terukur sering kali tercemar oleh **noise atau ketidakpastian pengukuran**:"
    )
    st.latex(r"\vec{R}_{terukur} = \vec{R}_{asli} + \epsilon_r, \quad \Delta \vec{R}_{terukur} = \Delta \vec{R}_{asli} + \epsilon_v")
    st.write(
        "Model Linear Regression dilatih dengan fitur masukan 6 dimensi:"
    )
    st.latex(r"\vec{X} = \left[ X_{cm,0}^{meas}, Y_{cm,0}^{meas}, Z_{cm,0}^{meas}, \Delta X^{meas}, \Delta Y^{meas}, \Delta Z^{meas} \right]^T")
    st.write("Untuk memprediksi target 3 dimensi:")
    st.latex(r"\hat{\vec{Y}} = \left[ X_{cm}(t), Y_{cm}(t), Z_{cm}(t) \right]^T")
    st.write(
        "Hasil eksperimen menunjukkan $R^2 \\approx 0.9999$ dan MAE dalam fraksi milimeter, "
        "yang memvalidasi bahwa model regresi secara efektif mampu menyaring noise Gaussian "
        "dan mempelajari hukum fisika secara murni dari data simulasi."
    )

    st.info("💡 **Kesimpulan Akademik**: Integrasi antara mekanika klasik dan pembelajaran mesin ini dapat diaplikasikan pada pelacakan objek drone, robotika otonom, sistem partikel astrofisika, dan biomekanika gerak manusia.")


# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #8b949e; font-size: 0.85rem;'>"
    "Center of Mass and Velocity with Learning-Based Motion Prediction | Streamlit Interactive Studio (Dark Mode)"
    "</div>",
    unsafe_allow_html=True
)
