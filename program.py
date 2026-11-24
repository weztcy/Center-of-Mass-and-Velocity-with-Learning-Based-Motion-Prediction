# -*- coding: utf-8 -*-
"""
PROJECT: Center of Mass + Velocity with Machine Learning
Algoritma ML: Linear Regression
Dataset: Synthetic Dataset (dibuat otomatis dari simulasi fisika)

Instalasi library:
    pip install numpy matplotlib pandas scikit-learn

Jalankan:
    python Center_of_Mass_Velocity_ML.py

Catatan akademik:
Model mempelajari hubungan antara posisi pusat massa awal dan perpindahan
pusat massa untuk memperkirakan posisi pusat massa di masa depan.
Dataset dibuat dari simulasi gerak berkecepatan konstan. Sedikit noise pada
fitur input digunakan untuk meniru ketidakpastian pengukuran.
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# BAGIAN 1 - PERHITUNGAN FISIKA (PROGRAM ASLI)
# ============================================================

def hitung_pusat_massa(posisi, massa):
    """Menghitung pusat massa dari beberapa objek dalam 3 dimensi."""
    posisi = np.asarray(posisi, dtype=float)
    massa = np.asarray(massa, dtype=float)

    if posisi.ndim != 2 or posisi.shape[1] != 3:
        raise ValueError("Posisi harus berupa array dengan 3 kolom (X, Y, Z).")
    if massa.ndim != 1 or len(massa) != len(posisi):
        raise ValueError("Jumlah massa harus sama dengan jumlah objek.")
    if np.any(massa <= 0):
        raise ValueError("Massa setiap objek harus lebih besar dari nol.")

    total_massa = np.sum(massa)
    return np.sum(posisi * massa[:, np.newaxis], axis=0) / total_massa


def hitung_kecepatan_pusat_massa(kecepatan, massa):
    """Menghitung kecepatan pusat massa dalam 3 dimensi."""
    kecepatan = np.asarray(kecepatan, dtype=float)
    massa = np.asarray(massa, dtype=float)

    if kecepatan.ndim != 2 or kecepatan.shape[1] != 3:
        raise ValueError("Kecepatan harus berupa array dengan 3 kolom (Vx, Vy, Vz).")
    if massa.ndim != 1 or len(massa) != len(kecepatan):
        raise ValueError("Jumlah massa harus sama dengan jumlah objek.")
    if np.any(massa <= 0):
        raise ValueError("Massa setiap objek harus lebih besar dari nol.")

    total_massa = np.sum(massa)
    return np.sum(kecepatan * massa[:, np.newaxis], axis=0) / total_massa


# ============================================================
# BAGIAN 2 - VISUALISASI PROGRAM ASLI
# ============================================================

def visualisasi_2d(posisi, pusat_massa, judul, simpan_path=None):
    plt.figure(figsize=(7, 5))
    warna = plt.cm.viridis(np.linspace(0, 1, len(posisi)))
    for i, pos in enumerate(posisi):
        plt.scatter(pos[0], pos[1], color=warna[i], label=f"Objek {i + 1}")
    plt.scatter(pusat_massa[0], pusat_massa[1], color="red", edgecolor="black",
                s=120, label="Pusat Massa")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(judul)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if simpan_path:
        plt.savefig(simpan_path, dpi=300, bbox_inches="tight")
        print(f"Visualisasi disimpan ke: {simpan_path}")
    plt.show()


def visualisasi_3d(posisi, pusat_massa, judul, simpan_path=None):
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")
    warna = plt.cm.viridis(np.linspace(0, 1, len(posisi)))
    for i, pos in enumerate(posisi):
        ax.scatter(pos[0], pos[1], pos[2], color=warna[i], label=f"Objek {i + 1}")
    ax.scatter(pusat_massa[0], pusat_massa[1], pusat_massa[2], color="red",
               edgecolor="black", s=120, label="Pusat Massa")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(judul)
    ax.legend()
    plt.tight_layout()
    if simpan_path:
        plt.savefig(simpan_path, dpi=300, bbox_inches="tight")
        print(f"Visualisasi disimpan ke: {simpan_path}")
    plt.show()


def visualisasi_kecepatan_2d(kecepatan, kecepatan_pusat_massa, judul, simpan_path=None):
    plt.figure(figsize=(7, 5))
    warna = plt.cm.viridis(np.linspace(0, 1, len(kecepatan)))
    for i, v in enumerate(kecepatan):
        plt.quiver(0, 0, v[0], v[1], color=warna[i], scale=1,
                   scale_units="xy", angles="xy", label=f"Objek {i + 1}")
    plt.quiver(0, 0, kecepatan_pusat_massa[0], kecepatan_pusat_massa[1],
               color="red", scale=1, scale_units="xy", angles="xy",
               label="Kecepatan Pusat Massa")
    plt.xlabel("Vx")
    plt.ylabel("Vy")
    plt.title(judul)
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.tight_layout()
    if simpan_path:
        plt.savefig(simpan_path, dpi=300, bbox_inches="tight")
        print(f"Visualisasi disimpan ke: {simpan_path}")
    plt.show()


def visualisasi_kecepatan_3d(kecepatan, kecepatan_pusat_massa, judul, simpan_path=None):
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")
    warna = plt.cm.viridis(np.linspace(0, 1, len(kecepatan)))
    for i, v in enumerate(kecepatan):
        ax.quiver(0, 0, 0, v[0], v[1], v[2], color=warna[i], label=f"Objek {i + 1}")
    ax.quiver(0, 0, 0, kecepatan_pusat_massa[0], kecepatan_pusat_massa[1],
              kecepatan_pusat_massa[2], color="red",
              label="Kecepatan Pusat Massa")
    ax.set_xlabel("Vx")
    ax.set_ylabel("Vy")
    ax.set_zlabel("Vz")
    ax.set_title(judul)
    ax.legend()
    plt.tight_layout()
    if simpan_path:
        plt.savefig(simpan_path, dpi=300, bbox_inches="tight")
        print(f"Visualisasi disimpan ke: {simpan_path}")
    plt.show()


# ============================================================
# BAGIAN 3 - MEMBUAT DATASET SINTETIS
# ============================================================

FEATURE_COLUMNS = [
    "cm_x_awal_terukur", "cm_y_awal_terukur", "cm_z_awal_terukur",
    "perpindahan_x_terukur", "perpindahan_y_terukur", "perpindahan_z_terukur",
]
TARGET_COLUMNS = [
    "cm_x_masa_depan", "cm_y_masa_depan", "cm_z_masa_depan",
]


def buat_dataset_sintetis(jumlah_sampel=3000, noise_std=0.05, random_state=42):
    """Membuat data dari simulasi gerak beberapa objek."""
    rng = np.random.default_rng(random_state)
    baris_dataset = []

    for _ in range(jumlah_sampel):
        jumlah_objek = int(rng.integers(2, 7))
        massa = rng.uniform(0.5, 10.0, size=jumlah_objek)
        posisi_awal = rng.uniform(-20.0, 20.0, size=(jumlah_objek, 3))
        kecepatan = rng.uniform(-5.0, 5.0, size=(jumlah_objek, 3))
        waktu = float(rng.uniform(0.1, 5.0))

        cm_awal_asli = hitung_pusat_massa(posisi_awal, massa)
        v_cm = hitung_kecepatan_pusat_massa(kecepatan, massa)
        perpindahan_asli = v_cm * waktu
        cm_masa_depan = cm_awal_asli + perpindahan_asli

        # Noise pada fitur input meniru ketidakpastian pengukuran.
        cm_awal_terukur = cm_awal_asli + rng.normal(0, noise_std, size=3)
        perpindahan_terukur = perpindahan_asli + rng.normal(0, noise_std, size=3)

        baris_dataset.append({
            "jumlah_objek": jumlah_objek,
            "waktu_s": waktu,
            "kecepatan_cm_x": v_cm[0],
            "kecepatan_cm_y": v_cm[1],
            "kecepatan_cm_z": v_cm[2],
            "cm_x_awal_terukur": cm_awal_terukur[0],
            "cm_y_awal_terukur": cm_awal_terukur[1],
            "cm_z_awal_terukur": cm_awal_terukur[2],
            "perpindahan_x_terukur": perpindahan_terukur[0],
            "perpindahan_y_terukur": perpindahan_terukur[1],
            "perpindahan_z_terukur": perpindahan_terukur[2],
            "cm_x_masa_depan": cm_masa_depan[0],
            "cm_y_masa_depan": cm_masa_depan[1],
            "cm_z_masa_depan": cm_masa_depan[2],
        })

    return pd.DataFrame(baris_dataset)


# ============================================================
# BAGIAN 4 - TRAINING DAN EVALUASI LINEAR REGRESSION
# ============================================================

def visualisasi_hasil_evaluasi(y_aktual, y_prediksi, simpan_path=None):
    aktual = y_aktual.to_numpy()
    nama_sumbu = ["X", "Y", "Z"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for i, ax in enumerate(axes):
        ax.scatter(aktual[:, i], y_prediksi[:, i], alpha=0.5)
        nilai_min = min(aktual[:, i].min(), y_prediksi[:, i].min())
        nilai_max = max(aktual[:, i].max(), y_prediksi[:, i].max())
        ax.plot([nilai_min, nilai_max], [nilai_min, nilai_max], "r--")
        ax.set_xlabel(f"Nilai Aktual {nama_sumbu[i]}")
        ax.set_ylabel(f"Prediksi {nama_sumbu[i]}")
        ax.set_title(f"Prediksi Sumbu {nama_sumbu[i]}")
        ax.grid(True)

    fig.suptitle("Perbandingan Nilai Aktual dan Prediksi Linear Regression")
    plt.tight_layout()
    if simpan_path:
        plt.savefig(simpan_path, dpi=300, bbox_inches="tight")
        print(f"Visualisasi disimpan ke: {simpan_path}")
    plt.show()


def visualisasi_prediksi_3d(cm_awal, prediksi_fisika, prediksi_ml, simpan_path=None):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(*cm_awal, color="blue", s=90, label="Pusat Massa Awal")
    ax.scatter(*prediksi_fisika, color="green", marker="^", s=110,
               label="Hasil Rumus Fisika")
    ax.scatter(*prediksi_ml, color="red", marker="x", s=110,
               label="Prediksi Machine Learning")
    ax.plot([cm_awal[0], prediksi_fisika[0]],
            [cm_awal[1], prediksi_fisika[1]],
            [cm_awal[2], prediksi_fisika[2]], "k--", alpha=0.6,
            label="Arah perpindahan")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Perbandingan Prediksi Posisi Pusat Massa")
    ax.legend()
    plt.tight_layout()
    if simpan_path:
        plt.savefig(simpan_path, dpi=300, bbox_inches="tight")
        print(f"Visualisasi disimpan ke: {simpan_path}")
    plt.show()


def baca_float(pesan, harus_positif=False):
    while True:
        try:
            nilai = float(input(pesan))
            if not np.isfinite(nilai):
                print("Masukkan angka yang valid.")
                continue
            if harus_positif and nilai <= 0:
                print("Nilai harus lebih besar dari nol.")
                continue
            return nilai
        except ValueError:
            print("Input harus berupa angka.")


def baca_jumlah_objek():
    while True:
        try:
            jumlah = int(input("Masukkan jumlah objek: "))
            if jumlah < 1:
                print("Jumlah objek minimal 1.")
                continue
            return jumlah
        except ValueError:
            print("Jumlah objek harus berupa bilangan bulat.")


def prediksi_dengan_data_sendiri(model):
    print("\n--- Masukkan data untuk prediksi ---")
    jumlah_objek = baca_jumlah_objek()
    massa, posisi, kecepatan = [], [], []

    for i in range(jumlah_objek):
        print(f"\nObjek ke-{i + 1}")
        m = baca_float("Massa: ", harus_positif=True)
        x = baca_float("Posisi X: ")
        y = baca_float("Posisi Y: ")
        z = baca_float("Posisi Z: ")
        vx = baca_float("Kecepatan Vx: ")
        vy = baca_float("Kecepatan Vy: ")
        vz = baca_float("Kecepatan Vz: ")
        massa.append(m)
        posisi.append([x, y, z])
        kecepatan.append([vx, vy, vz])

    waktu = baca_float("\nWaktu prediksi (detik): ", harus_positif=True)
    massa = np.asarray(massa, dtype=float)
    posisi = np.asarray(posisi, dtype=float)
    kecepatan = np.asarray(kecepatan, dtype=float)

    cm_awal = hitung_pusat_massa(posisi, massa)
    v_cm = hitung_kecepatan_pusat_massa(kecepatan, massa)
    perpindahan = v_cm * waktu

    fitur_baru = pd.DataFrame([[*cm_awal, *perpindahan]], columns=FEATURE_COLUMNS)
    prediksi_ml = model.predict(fitur_baru)[0]
    prediksi_fisika = cm_awal + perpindahan

    print("\n--- Hasil prediksi ---")
    print("Pusat massa awal (X, Y, Z):", np.round(cm_awal, 4))
    print("Kecepatan pusat massa (Vx, Vy, Vz):", np.round(v_cm, 4))
    print("Hasil rumus fisika (X, Y, Z):", np.round(prediksi_fisika, 4))
    print("Prediksi Machine Learning (X, Y, Z):", np.round(prediksi_ml, 4))
    print("Selisih absolut ML terhadap fisika:",
          np.round(np.abs(prediksi_ml - prediksi_fisika), 4))
    visualisasi_prediksi_3d(cm_awal, prediksi_fisika, prediksi_ml)


def jalankan_machine_learning():
    print("\n=== MACHINE LEARNING: LINEAR REGRESSION ===")
    print("Membuat 3.000 skenario dataset sintetis...")

    dataset = buat_dataset_sintetis(jumlah_sampel=3000)
    nama_file = "dataset_center_of_mass_ml.csv"
    dataset.to_csv(nama_file, index=False)
    print(f"Dataset berhasil dibuat: {nama_file}")
    print("Jumlah baris dataset:", len(dataset))
    print("\nContoh 5 baris dataset:")
    print(dataset.head())

    X = dataset[FEATURE_COLUMNS]
    y = dataset[TARGET_COLUMNS]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_prediksi = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_prediksi)
    rmse = np.sqrt(mean_squared_error(y_test, y_prediksi))
    r2 = r2_score(y_test, y_prediksi)

    print("\n--- Hasil Evaluasi Model ---")
    print(f"Data training : {len(X_train)}")
    print(f"Data testing  : {len(X_test)}")
    print(f"MAE           : {mae:.6f}")
    print(f"RMSE          : {rmse:.6f}")
    print(f"R-squared     : {r2:.6f}")
    print("\nMAE dan RMSE yang lebih kecil menunjukkan kesalahan prediksi lebih kecil.")
    print("R-squared yang mendekati 1 menunjukkan kecocokan tinggi pada data uji.")

    visualisasi_hasil_evaluasi(y_test, y_prediksi)

    jawaban = input(
        "\nApakah ingin mencoba prediksi dengan data objek sendiri? (y/n): "
    ).strip().lower()
    if jawaban == "y":
        prediksi_dengan_data_sendiri(model)

    print("\nTraining dan evaluasi Machine Learning selesai.")


# ============================================================
# BAGIAN 5 - MENU UTAMA
# ============================================================

def main():
    if "--export" in sys.argv:
        from extract_visualizations import main as jalankan_ekspor
        jalankan_ekspor()
        return

    while True:
        print("\n==========================================")
        print("       CENTER OF MASS + VELOCITY")
        print("==========================================")
        print("1. Hitung Pusat Massa")
        print("2. Hitung Kecepatan Pusat Massa")
        print("3. Machine Learning - Prediksi Posisi CM")
        print("4. Ekspor Semua Visualisasi (High-Res PNG)")
        print("5. Keluar")
        pilihan = input("Pilih opsi (1/2/3/4/5): ").strip()

        if pilihan == "1":
            jumlah_objek = baca_jumlah_objek()
            massa, posisi = [], []
            for i in range(jumlah_objek):
                print(f"\nObjek ke-{i + 1}")
                massa.append(baca_float("Massa: ", harus_positif=True))
                x = baca_float("Koordinat X: ")
                y = baca_float("Koordinat Y: ")
                z = baca_float("Koordinat Z: ")
                posisi.append([x, y, z])
            posisi = np.asarray(posisi, dtype=float)
            massa = np.asarray(massa, dtype=float)
            cm = hitung_pusat_massa(posisi, massa)
            print("\nPusat Massa (X, Y, Z):", np.round(cm, 4))
            visualisasi_2d(posisi, cm, "Pusat Massa 2D")
            visualisasi_3d(posisi, cm, "Pusat Massa 3D")

        elif pilihan == "2":
            jumlah_objek = baca_jumlah_objek()
            massa, kecepatan = [], []
            for i in range(jumlah_objek):
                print(f"\nObjek ke-{i + 1}")
                massa.append(baca_float("Massa: ", harus_positif=True))
                vx = baca_float("Kecepatan Vx: ")
                vy = baca_float("Kecepatan Vy: ")
                vz = baca_float("Kecepatan Vz: ")
                kecepatan.append([vx, vy, vz])
            kecepatan = np.asarray(kecepatan, dtype=float)
            massa = np.asarray(massa, dtype=float)
            v_cm = hitung_kecepatan_pusat_massa(kecepatan, massa)
            print("\nKecepatan Pusat Massa (Vx, Vy, Vz):", np.round(v_cm, 4))
            visualisasi_kecepatan_2d(kecepatan, v_cm, "Kecepatan Pusat Massa 2D")
            visualisasi_kecepatan_3d(kecepatan, v_cm, "Kecepatan Pusat Massa 3D")

        elif pilihan == "3":
            jalankan_machine_learning()

        elif pilihan == "4":
            print("\nMemulai proses ekspor seluruh visualisasi ke format PNG 300 DPI...")
            from extract_visualizations import main as jalankan_ekspor
            jalankan_ekspor()

        elif pilihan == "5":
            print("Terima kasih! Program selesai.")
            break

        else:
            print("Opsi tidak valid. Silakan pilih 1, 2, 3, 4, atau 5.")


if __name__ == "__main__":
    main()
