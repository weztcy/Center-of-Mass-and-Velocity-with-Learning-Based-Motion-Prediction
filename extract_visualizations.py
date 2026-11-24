# -*- coding: utf-8 -*-
"""
Script untuk mengekstrak dan menyimpan seluruh visualisasi dari program
Center of Mass and Velocity with Learning-Based Motion Prediction
dalam format PNG resolusi tinggi (High Resolution / 300 DPI).
"""

import os
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend untuk ekspor gambar bersih
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Direktori penyimpanan visualisasi
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "visualizations")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Konfigurasi style Matplotlib untuk publikasi / laporan akademik
plt.rcParams.update({
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
    "font.family": "sans-serif",
    "figure.titlesize": 14,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# ============================================================
# 1. FUNGSI FISIKA
# ============================================================

def hitung_pusat_massa(posisi, massa):
    posisi = np.asarray(posisi, dtype=float)
    massa = np.asarray(massa, dtype=float)
    return np.sum(posisi * massa[:, np.newaxis], axis=0) / np.sum(massa)

def hitung_kecepatan_pusat_massa(kecepatan, massa):
    kecepatan = np.asarray(kecepatan, dtype=float)
    massa = np.asarray(massa, dtype=float)
    return np.sum(kecepatan * massa[:, np.newaxis], axis=0) / np.sum(massa)

# ============================================================
# 2. GENERASI VISUALISASI
# ============================================================

def export_visualisasi_2d_3d_cm(posisi, massa, cm):
    """Ekspor visualisasi Pusat Massa 2D dan 3D."""
    warna = plt.cm.plasma(np.linspace(0.1, 0.85, len(posisi)))
    
    # --- 1. Visualisasi 2D ---
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    for i, pos in enumerate(posisi):
        ax.scatter(pos[0], pos[1], color=warna[i], s=100 + massa[i] * 35,
                   alpha=0.85, edgecolors='black', linewidth=1.2,
                   label=f"Objek {i + 1} (m={massa[i]} kg)")
        ax.annotate(f"Objek {i+1}\n({pos[0]:.1f}, {pos[1]:.1f})",
                    (pos[0], pos[1]), textcoords="offset points",
                    xytext=(8, 8), fontsize=9, fontweight='medium')

    ax.scatter(cm[0], cm[1], color="crimson", edgecolor="black", linewidth=1.5,
               s=220, marker="*", label=f"Pusat Massa ({cm[0]:.2f}, {cm[1]:.2f})", zorder=5)
    
    # Garis bantu proyeksi
    ax.axvline(cm[0], color="crimson", linestyle=":", alpha=0.5)
    ax.axhline(cm[1], color="crimson", linestyle=":", alpha=0.5)
    
    ax.set_title("Visualisasi Pusat Massa dalam Bidang 2D (Sumbu X-Y)", fontweight="bold", pad=12)
    ax.set_xlabel("Koordinat X (m)", fontweight="semibold")
    ax.set_ylabel("Koordinat Y (m)", fontweight="semibold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, facecolor="white", edgecolor="none", shadow=True, loc="upper right")
    
    file_path_2d = os.path.join(OUTPUT_DIR, "01_pusat_massa_2d.png")
    fig.savefig(file_path_2d, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: {file_path_2d}")

    # --- 2. Visualisasi 3D ---
    fig = plt.figure(figsize=(9, 7), dpi=300)
    ax = fig.add_subplot(111, projection="3d")
    
    for i, pos in enumerate(posisi):
        ax.scatter(pos[0], pos[1], pos[2], color=warna[i], s=100 + massa[i] * 35,
                   alpha=0.9, edgecolors='black', linewidth=1.0,
                   label=f"Objek {i + 1} (m={massa[i]} kg)")
        # Garis proyeksi ke dasar z_min
        ax.plot([pos[0], pos[0]], [pos[1], pos[1]], [-5, pos[2]], color=warna[i], linestyle=":", alpha=0.4)

    ax.scatter(cm[0], cm[1], cm[2], color="crimson", edgecolor="black", linewidth=1.5,
               s=260, marker="*", label=f"Pusat Massa CM ({cm[0]:.2f}, {cm[1]:.2f}, {cm[2]:.2f})", zorder=10)
    ax.plot([cm[0], cm[0]], [cm[1], cm[1]], [-5, cm[2]], color="crimson", linestyle="--", alpha=0.7)

    ax.set_title("Visualisasi Pusat Massa Sistem Multi-Objek dalam Ruang 3D", fontweight="bold", pad=14)
    ax.set_xlabel("Sumbu X (m)", fontweight="semibold", labelpad=8)
    ax.set_ylabel("Sumbu Y (m)", fontweight="semibold", labelpad=8)
    ax.set_zlabel("Sumbu Z (m)", fontweight="semibold", labelpad=8)
    ax.view_init(elev=24, azim=40)
    ax.legend(frameon=True, loc="upper left", bbox_to_anchor=(0.0, 0.95))
    
    file_path_3d = os.path.join(OUTPUT_DIR, "02_pusat_massa_3d.png")
    fig.savefig(file_path_3d, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: {file_path_3d}")


def export_visualisasi_kecepatan(kecepatan, massa, v_cm):
    """Ekspor visualisasi Vektor Kecepatan 2D dan 3D."""
    warna = plt.cm.viridis(np.linspace(0.15, 0.85, len(kecepatan)))
    
    # --- 3. Kecepatan 2D (Quiver) ---
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    
    for i, v in enumerate(kecepatan):
        ax.quiver(0, 0, v[0], v[1], color=warna[i], angles="xy", scale_units="xy", scale=1,
                  alpha=0.85, width=0.007,
                  label=f"v Objek {i + 1}: ({v[0]:.1f}, {v[1]:.1f}) m/s")

    ax.quiver(0, 0, v_cm[0], v_cm[1], color="crimson", angles="xy", scale_units="xy", scale=1,
              width=0.012, zorder=6,
              label=f"v Pusat Massa: ({v_cm[0]:.2f}, {v_cm[1]:.2f}) m/s")

    # Batas plot proporsional
    max_val = max(np.max(np.abs(kecepatan[:, :2])), np.max(np.abs(v_cm[:2]))) * 1.3
    ax.set_xlim(-max_val, max_val)
    ax.set_ylim(-max_val, max_val)
    ax.set_aspect("equal", adjustable="box")
    
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="-", alpha=0.5)
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="-", alpha=0.5)
    
    ax.set_title("Diagram Vektor Kecepatan Objek & Kecepatan Pusat Massa (2D)", fontweight="bold", pad=12)
    ax.set_xlabel("Kecepatan Vx (m/s)", fontweight="semibold")
    ax.set_ylabel("Kecepatan Vy (m/s)", fontweight="semibold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, facecolor="white", loc="best", shadow=True)

    file_path_v2d = os.path.join(OUTPUT_DIR, "03_kecepatan_pusat_massa_2d.png")
    fig.savefig(file_path_v2d, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: {file_path_v2d}")

    # --- 4. Kecepatan 3D (Quiver) ---
    fig = plt.figure(figsize=(9, 7), dpi=300)
    ax = fig.add_subplot(111, projection="3d")
    
    for i, v in enumerate(kecepatan):
        ax.quiver(0, 0, 0, v[0], v[1], v[2], color=warna[i], linewidth=2.0, alpha=0.85,
                  arrow_length_ratio=0.15, label=f"v Objek {i + 1} ({v[0]:.1f}, {v[1]:.1f}, {v[2]:.1f})")

    ax.quiver(0, 0, 0, v_cm[0], v_cm[1], v_cm[2], color="crimson", linewidth=3.5,
              arrow_length_ratio=0.15,
              label=f"v Pusat Massa ({v_cm[0]:.2f}, {v_cm[1]:.2f}, {v_cm[2]:.2f})")

    max_v3d = max(np.max(np.abs(kecepatan)), np.max(np.abs(v_cm))) * 1.2
    ax.set_xlim([-max_v3d, max_v3d])
    ax.set_ylim([-max_v3d, max_v3d])
    ax.set_zlim([-max_v3d, max_v3d])
    
    ax.set_title("Vektor Kecepatan Multi-Objek dan Pusat Massa dalam Ruang 3D", fontweight="bold", pad=14)
    ax.set_xlabel("Vx (m/s)", fontweight="semibold", labelpad=8)
    ax.set_ylabel("Vy (m/s)", fontweight="semibold", labelpad=8)
    ax.set_zlabel("Vz (m/s)", fontweight="semibold", labelpad=8)
    ax.view_init(elev=20, azim=55)
    ax.legend(frameon=True, loc="upper left", bbox_to_anchor=(0.0, 0.98))

    file_path_v3d = os.path.join(OUTPUT_DIR, "04_kecepatan_pusat_massa_3d.png")
    fig.savefig(file_path_v3d, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: {file_path_v3d}")


def export_evaluasi_ml(y_test, y_prediksi, mae_list, rmse_list, r2_list):
    """Ekspor visualisasi evaluasi performa Linear Regression."""
    aktual = y_test.to_numpy()
    sumbu = ["X", "Y", "Z"]
    colors = ["#2b5c8f", "#2a9d8f", "#e76f51"]

    # --- 5. Evaluasi Aktual vs Prediksi (3 Sumbu) ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=300)
    
    for i, ax in enumerate(axes):
        ax.scatter(aktual[:, i], y_prediksi[:, i], alpha=0.45, color=colors[i], edgecolors="none", s=25)
        
        nilai_min = min(aktual[:, i].min(), y_prediksi[:, i].min())
        nilai_max = max(aktual[:, i].max(), y_prediksi[:, i].max())
        ax.plot([nilai_min, nilai_max], [nilai_min, nilai_max], color="crimson", linestyle="--", linewidth=1.8, label="Ideal (y = x)")
        
        # Info Box Performa
        info_text = f"$R^2$: {r2_list[i]:.5f}\nMAE: {mae_list[i]:.4f}\nRMSE: {rmse_list[i]:.4f}"
        ax.text(0.05, 0.93, info_text, transform=ax.transAxes, verticalalignment="top",
                fontsize=9.5, bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#cccccc", alpha=0.9))

        ax.set_title(f"Sumbu {sumbu[i]} (cm_{sumbu[i].lower()}_masa_depan)", fontweight="bold", pad=8)
        ax.set_xlabel(f"Nilai Aktual {sumbu[i]} (m)", fontweight="semibold")
        ax.set_ylabel(f"Nilai Prediksi {sumbu[i]} (m)", fontweight="semibold")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="lower right")

    fig.suptitle("Evaluasi Model Linear Regression: Nilai Aktual vs Prediksi Posisi Pusat Massa",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    file_path_eval = os.path.join(OUTPUT_DIR, "05_evaluasi_linear_regression.png")
    fig.savefig(file_path_eval, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: {file_path_eval}")

    # --- 6. Distribusi Residual / Error ---
    residuals = y_prediksi - aktual
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), dpi=300)
    for i, ax in enumerate(axes):
        res = residuals[:, i]
        ax.hist(res, bins=35, color=colors[i], alpha=0.75, edgecolor="black", linewidth=0.8, density=True)
        ax.axvline(0, color="crimson", linestyle="--", linewidth=1.5, label="Error = 0")
        ax.axvline(np.mean(res), color="black", linestyle=":", linewidth=1.2, label=f"Mean: {np.mean(res):.4f}")
        
        ax.set_title(f"Distribusi Residual Sumbu {sumbu[i]} ($\\Delta {sumbu[i]}$)", fontweight="bold")
        ax.set_xlabel(f"Residual Prediksi (m)", fontweight="semibold")
        ax.set_ylabel("Kerapatan (Density)", fontweight="semibold")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="upper right", fontsize=8.5)

    fig.suptitle("Distribusi Residual (Prediksi - Aktual) Model Linear Regression",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    file_path_res = os.path.join(OUTPUT_DIR, "06_distribusi_residual_error.png")
    fig.savefig(file_path_res, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: {file_path_res}")


def export_prediksi_posisi_3d(cm_awal, prediksi_fisika, prediksi_ml, trajectory_t=None):
    """Ekspor visualisasi komparasi prediksi posisi masa depan (Fisika vs ML)."""
    fig = plt.figure(figsize=(9, 7), dpi=300)
    ax = fig.add_subplot(111, projection="3d")
    
    # Titik Pusat Massa Awal
    ax.scatter(*cm_awal, color="#1d3557", s=140, marker="o", edgecolors="black",
               label=f"Posisi CM Awal t=0 ({cm_awal[0]:.2f}, {cm_awal[1]:.2f}, {cm_awal[2]:.2f})", zorder=5)
    
    # Prediksi Fisika
    ax.scatter(*prediksi_fisika, color="#2a9d8f", marker="^", s=160, edgecolors="black",
               label=f"Hasil Rumus Fisika t=2s ({prediksi_fisika[0]:.2f}, {prediksi_fisika[1]:.2f}, {prediksi_fisika[2]:.2f})", zorder=6)
    
    # Prediksi ML
    ax.scatter(*prediksi_ml, color="#e63946", marker="X", s=160, edgecolors="black",
               label=f"Prediksi Machine Learning ({prediksi_ml[0]:.2f}, {prediksi_ml[1]:.2f}, {prediksi_ml[2]:.2f})", zorder=7)

    # Garis Vektor Lintasan Fisika
    ax.plot([cm_awal[0], prediksi_fisika[0]],
            [cm_awal[1], prediksi_fisika[1]],
            [cm_awal[2], prediksi_fisika[2]],
            color="#2a9d8f", linestyle="--", linewidth=2.0, alpha=0.8,
            label="Trajektori Fisika Konstan")
    
    # Garis Hubung Prediksi ML
    ax.plot([cm_awal[0], prediksi_ml[0]],
            [cm_awal[1], prediksi_ml[1]],
            [cm_awal[2], prediksi_ml[2]],
            color="#e63946", linestyle=":", linewidth=1.8, alpha=0.7,
            label="Arah Prediksi ML")

    # Selisih/Error Line antara Fisika dan ML
    ax.plot([prediksi_fisika[0], prediksi_ml[0]],
            [prediksi_fisika[1], prediksi_ml[1]],
            [prediksi_fisika[2], prediksi_ml[2]],
            color="black", linestyle="-", linewidth=1.5,
            label=f"Deviasi ML-Fisika ({np.linalg.norm(prediksi_ml - prediksi_fisika):.4f} m)")

    ax.set_title("Perbandingan Posisi Pusat Massa: Rumus Fisika vs Machine Learning", fontweight="bold", pad=14)
    ax.set_xlabel("Sumbu X (m)", fontweight="semibold", labelpad=8)
    ax.set_ylabel("Sumbu Y (m)", fontweight="semibold", labelpad=8)
    ax.set_zlabel("Sumbu Z (m)", fontweight="semibold", labelpad=8)
    ax.view_init(elev=22, azim=48)
    ax.legend(frameon=True, loc="upper left", bbox_to_anchor=(0.0, 0.98), fontsize=9)

    file_path_pred = os.path.join(OUTPUT_DIR, "07_perbandingan_prediksi_posisi_3d.png")
    fig.savefig(file_path_pred, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: {file_path_pred}")


def export_dashboard_lengkap(posisi, massa, cm, kecepatan, v_cm, y_test, y_prediksi, cm_awal, prediksi_fisika, prediksi_ml, r2_val, mae_val, rmse_val):
    """Ekspor satu dashboard komprehensif 2x2 yang memadukan seluruh pilar proyek."""
    fig = plt.figure(figsize=(18, 14), dpi=300)
    
    # Subplot 1: Pusat Massa 2D
    ax1 = fig.add_subplot(2, 2, 1)
    warna1 = plt.cm.plasma(np.linspace(0.1, 0.85, len(posisi)))
    for i, pos in enumerate(posisi):
        ax1.scatter(pos[0], pos[1], color=warna1[i], s=80 + massa[i] * 25,
                    alpha=0.85, edgecolors='black', label=f"Objek {i+1} ({massa[i]} kg)")
    ax1.scatter(cm[0], cm[1], color="crimson", edgecolor="black", s=180, marker="*",
                label=f"CM ({cm[0]:.2f}, {cm[1]:.2f})", zorder=5)
    ax1.set_title("(A) Distribusi Objek & Pusat Massa (2D)", fontweight="bold")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", fontsize=8.5)

    # Subplot 2: Kecepatan Pusat Massa 2D
    ax2 = fig.add_subplot(2, 2, 2)
    warna2 = plt.cm.viridis(np.linspace(0.15, 0.85, len(kecepatan)))
    for i, v in enumerate(kecepatan):
        ax2.quiver(0, 0, v[0], v[1], color=warna2[i], angles="xy", scale_units="xy", scale=1,
                   alpha=0.85, width=0.007, label=f"v Objek {i+1}")
    ax2.quiver(0, 0, v_cm[0], v_cm[1], color="crimson", angles="xy", scale_units="xy", scale=1,
               width=0.012, zorder=6, label=f"v_CM ({v_cm[0]:.2f}, {v_cm[1]:.2f}) m/s")
    max_val = max(np.max(np.abs(kecepatan[:, :2])), np.max(np.abs(v_cm[:2]))) * 1.3
    ax2.set_xlim(-max_val, max_val)
    ax2.set_ylim(-max_val, max_val)
    ax2.set_aspect("equal", adjustable="box")
    ax2.set_title("(B) Vektor Kecepatan Objek & Kecepatan CM (2D)", fontweight="bold")
    ax2.set_xlabel("Vx (m/s)")
    ax2.set_ylabel("Vy (m/s)")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right", fontsize=8.5)

    # Subplot 3: Evaluasi ML Aktual vs Prediksi (Sumbu X)
    ax3 = fig.add_subplot(2, 2, 3)
    aktual = y_test.to_numpy()
    ax3.scatter(aktual[:, 0], y_prediksi[:, 0], alpha=0.4, color="#2b5c8f", s=20)
    low_x = min(aktual[:, 0].min(), y_prediksi[:, 0].min())
    high_x = max(aktual[:, 0].max(), y_prediksi[:, 0].max())
    ax3.plot([low_x, high_x], [low_x, high_x], "r--", linewidth=1.5, label="y = x (Ideal)")
    ax3.text(0.05, 0.92, f"$R^2$: {r2_val:.5f}\nMAE: {mae_val:.4f} m\nRMSE: {rmse_val:.4f} m",
             transform=ax3.transAxes, verticalalignment="top", fontsize=9,
             bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#ddd"))
    ax3.set_title("(C) Evaluasi Model ML: Aktual vs Prediksi Sumbu X", fontweight="bold")
    ax3.set_xlabel("Nilai Aktual X (m)")
    ax3.set_ylabel("Nilai Prediksi X (m)")
    ax3.grid(True, linestyle="--", alpha=0.4)
    ax3.legend(loc="lower right", fontsize=8.5)

    # Subplot 4: 3D Prediksi Fisika vs ML
    ax4 = fig.add_subplot(2, 2, 4, projection="3d")
    ax4.scatter(*cm_awal, color="#1d3557", s=100, label="CM Awal (t=0)")
    ax4.scatter(*prediksi_fisika, color="#2a9d8f", marker="^", s=120, label="Rumus Fisika (t=2s)")
    ax4.scatter(*prediksi_ml, color="#e63946", marker="X", s=120, label="Prediksi ML (t=2s)")
    ax4.plot([cm_awal[0], prediksi_fisika[0]],
             [cm_awal[1], prediksi_fisika[1]],
             [cm_awal[2], prediksi_fisika[2]],
             color="#2a9d8f", linestyle="--", linewidth=1.8, label="Lintasan Fisika")
    ax4.set_title("(D) Trajektori Prediksi Posisi Pusat Massa 3D", fontweight="bold")
    ax4.set_xlabel("X (m)")
    ax4.set_ylabel("Y (m)")
    ax4.set_zlabel("Z (m)")
    ax4.view_init(elev=20, azim=45)
    ax4.legend(loc="upper left", fontsize=8)

    fig.suptitle("Pusat Massa, Kecepatan, dan Prediksi Gerak Berbasis Machine Learning",
                 fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    file_path_dash = os.path.join(OUTPUT_DIR, "08_dashboard_ringkasan_proyek.png")
    fig.savefig(file_path_dash, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: {file_path_dash}")


# ============================================================
# 3. PIPELINE UTAMA
# ============================================================

def main():
    print("=" * 60)
    print("  EKSTRAKSI VISUALISASI HIGH RESOLUTION (300 DPI)")
    print("=" * 60)

    # Skenario representatif multi-objek
    massa = np.array([2.5, 4.0, 3.2, 5.0])
    posisi = np.array([
        [2.0, 3.5, 1.2],
        [-3.0, 1.0, 4.0],
        [4.2, -2.5, 2.8],
        [-1.5, -3.2, -1.8]
    ])
    kecepatan = np.array([
        [2.0, 1.2, 0.5],
        [-1.2, 2.5, -1.0],
        [1.8, -1.5, 2.2],
        [-2.0, 0.8, 1.4]
    ])

    print("\n[1/4] Menghitung Fisika Pusat Massa & Kecepatan...")
    cm = hitung_pusat_massa(posisi, massa)
    v_cm = hitung_kecepatan_pusat_massa(kecepatan, massa)
    print(f"Pusat Massa (X, Y, Z)          : {np.round(cm, 4)}")
    print(f"Kecepatan Pusat Massa (Vx,Vy,Vz): {np.round(v_cm, 4)}")

    print("\n[2/4] Mengekspor Visualisasi Fisika 2D & 3D...")
    export_visualisasi_2d_3d_cm(posisi, massa, cm)
    export_visualisasi_kecepatan(kecepatan, massa, v_cm)

    print("\n[3/4] Melakukan Training & Evaluasi Model Machine Learning...")
    # Load dataset jika sudah ada, atau generate
    dataset_file = "dataset_center_of_mass_ml.csv"
    if os.path.exists(dataset_file):
        print(f"Memuat dataset yang sudah ada: {dataset_file}")
        dataset = pd.read_csv(dataset_file)
    else:
        print("Membuat dataset sintetis baru...")
        from program import buat_dataset_sintetis
        dataset = buat_dataset_sintetis(jumlah_sampel=3000)
        dataset.to_csv(dataset_file, index=False)

    FEATURE_COLUMNS = [
        "cm_x_awal_terukur", "cm_y_awal_terukur", "cm_z_awal_terukur",
        "perpindahan_x_terukur", "perpindahan_y_terukur", "perpindahan_z_terukur",
    ]
    TARGET_COLUMNS = [
        "cm_x_masa_depan", "cm_y_masa_depan", "cm_z_masa_depan",
    ]

    X = dataset[FEATURE_COLUMNS]
    y = dataset[TARGET_COLUMNS]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_prediksi = model.predict(X_test)

    # Metrik evaluasi per sumbu
    mae_list = [mean_absolute_error(y_test.iloc[:, i], y_prediksi[:, i]) for i in range(3)]
    rmse_list = [np.sqrt(mean_squared_error(y_test.iloc[:, i], y_prediksi[:, i])) for i in range(3)]
    r2_list = [r2_score(y_test.iloc[:, i], y_prediksi[:, i]) for i in range(3)]

    overall_mae = mean_absolute_error(y_test, y_prediksi)
    overall_rmse = np.sqrt(mean_squared_error(y_test, y_prediksi))
    overall_r2 = r2_score(y_test, y_prediksi)

    print(f"Hasil Evaluasi Keseluruhan:")
    print(f"  MAE  : {overall_mae:.6f} m")
    print(f"  RMSE : {overall_rmse:.6f} m")
    print(f"  R^2  : {overall_r2:.6f}")

    print("\n[4/4] Mengekspor Visualisasi Machine Learning & Prediksi Gerak...")
    export_evaluasi_ml(y_test, y_prediksi, mae_list, rmse_list, r2_list)

    # Skenario Prediksi Masa Depan (waktu = 2 detik)
    waktu = 2.0
    cm_awal_baru = cm
    perpindahan = v_cm * waktu
    fitur_baru = pd.DataFrame([[*cm_awal_baru, *perpindahan]], columns=FEATURE_COLUMNS)
    
    prediksi_ml = model.predict(fitur_baru)[0]
    prediksi_fisika = cm_awal_baru + perpindahan
    
    print("\nPrediksi Masa Depan (t = 2.0 s):")
    print("  Pusat Massa Awal    :", np.round(cm_awal_baru, 4))
    print("  Prediksi Fisika     :", np.round(prediksi_fisika, 4))
    print("  Prediksi ML         :", np.round(prediksi_ml, 4))
    print("  Selisih Deviasi     :", np.round(np.abs(prediksi_ml - prediksi_fisika), 4))

    export_prediksi_posisi_3d(cm_awal_baru, prediksi_fisika, prediksi_ml)
    export_dashboard_lengkap(posisi, massa, cm, kecepatan, v_cm, y_test, y_prediksi,
                             cm_awal_baru, prediksi_fisika, prediksi_ml,
                             overall_r2, overall_mae, overall_rmse)

    print("\n" + "=" * 60)
    print(f"SELESAI! Seluruh file PNG (300 DPI) tersimpan di:\n{OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
