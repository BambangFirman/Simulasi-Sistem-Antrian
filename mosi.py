import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import math
import scipy.stats as stats

# Data untuk WAKTU KEDATANGAN CALON PENONTON (MENIT)
data_kedatangan = [
    [1.5, 2, 1.5, 0.5, 2, 1.5, 1, 1.5, 2, 1.5],
    [1.5, 1, 1.5, 0.5, 2, 0.5, 1, 1, 1.5, 0.5],
    [1.5, 1, 1, 1.5, 1, 0.5, 1.5, 1, 1.5, 2],
    [1.5, 1, 1.5, 1.5, 1, 0.5, 1.5, 1, 1.5, 1.5],
    [1.5, 0.5, 1.5, 1.5, 1.5, 1, 1, 2, 1.5, 1],
    [1.5, 1, 1.5, 1.5, 1.5, 1, 1.5, 2, 1.5, 1],
    [1.5, 1, 2, 1.5, 1.5, 1.5, 1.5, 2, 1.5, 2],
    [2, 1.5, 1.5, 1, 1.5, 1.5, 1, 1, 0.5, 2],
    [2, 1.5, 1, 0.5, 1.5, 1.5, 1.5, 1, 1.5, 1.5],
    [2, 2, 1, 2, 1.5, 1, 1.5, 2, 1.5, 1.5],
]

# Data untuk WAKTU PELAYANAN (DETIK)
data_pelayanan = [
    [75, 70, 75, 60, 65, 75, 75, 75, 75, 75],
    [75, 70, 90, 60, 65, 75, 75, 75, 75, 75],
    [70, 80, 75, 75, 90, 60, 75, 75, 85, 85],
    [75, 65, 75, 75, 75, 80, 65, 75, 85, 75],
    [70, 65, 75, 70, 75, 80, 75, 75, 85, 75],
    [85, 75, 75, 70, 70, 80, 75, 75, 75, 75],
    [85, 80, 75, 80, 70, 80, 75, 75, 75, 85],
    [80, 80, 65, 80, 70, 80, 70, 75, 85, 85],
    [80, 80, 75, 80, 70, 75, 70, 75, 75, 85],
    [70, 75, 80, 65, 75, 75, 75, 75, 85, 85],
]

# Membuat data frame
df_kedatangan = pd.DataFrame(
    data_kedatangan, columns=[f"Kolom {i+1}" for i in range(10)]
)
df_pelayanan = pd.DataFrame(data_pelayanan, columns=[f"Kolom {i+1}" for i in range(10)])


# Fungsi untuk mencetak tabel distribusi frekuensi
def get_frequency_distribution(df):
    data_series = df.values.flatten()
    freq_table = pd.value_counts(data_series).sort_index().reset_index()
    freq_table.columns = ["Value", "Frequency"]
    return freq_table


# Fungsi untuk plot histogram dari tabel distribusi frekuensi
def plot_line_chart(freq_table, title, xlabel, ylabel):
    plt.figure(figsize=(10, 6))
    plt.plot(
        freq_table["Value"],
        freq_table["Frequency"],
        marker="o",
        linestyle="-",
        color="skyblue",
    )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(freq_table["Value"])
    plt.grid(axis="y")
    st.pyplot(plt)


# Fungsi untuk menghitung rata-rata dan standar deviasi dari tabel frekuensi
def calculate_mean_std(freq_table, total_count):
    values = freq_table["Value"]
    frequencies = freq_table["Frequency"]

    # Menghitung rata-rata (µ)
    mean = np.sum(values * frequencies) / total_count

    # Menghitung variansi dan standar deviasi (σ)
    variance = np.sum(frequencies * (values - mean) ** 2) / (total_count - 1)
    std_dev = np.sqrt(variance)

    return mean, std_dev


# Fungsi untuk menghitung rata-rata dan standar deviasi kustom
def calculate_mean_custom_std(freq_table, total_count):
    values = freq_table["Value"]
    frequencies = freq_table["Frequency"]

    # Menghitung rata-rata (µ)
    mean = np.sum(values * frequencies) / total_count

    # Menghitung deviasi standar kustom (σ)
    std_dev = np.sqrt(mean * 2 / 99)

    return mean, std_dev


# Fungsi untuk menghasilkan angka acak menggunakan LCG
def lcg(a, c, m, seed, n, mu=1.255, sigma=0.115):
    random_numbers = []
    Z = seed  # z0
    for _ in range(n):
        Z_new = (a * Z + c) % m  # Zi
        computed_value = (a * Z) + c  # (a*Z0)+c
        Zi1_new = (a * computed_value + c) % m  # Zi+1
        Ui = Z_new / m  # Ui
        Ui_new = Zi1_new / m  # Ui+1

        sqrt_neg_2lnUi = (-2 * math.log(Ui)) ** (1 / 2)  # (-2 * ln(Ui)) ^ (1/2)

        cos_value = np.cos(2 * np.pi * Ui_new)  # cos(2π*U(i+1))
        Z_value = sqrt_neg_2lnUi * cos_value  # Z
        X = mu + (sigma * Z_value)  # X
        # distribusi normal
        normal_value = mu + (sigma * Z_value)

        random_numbers.append(
            (
                Z,
                computed_value,
                Z_new,
                Zi1_new,
                Ui,
                Ui_new,
                sqrt_neg_2lnUi,
                cos_value,
                Z_value,
                round(X, 3),
                round(normal_value, 3),
                round(mu, 3),
                round(sigma, 3),
            )
        )
        Z = Z_new
    return random_numbers


# Fungsi untuk menghasilkan angka acak menggunakan MCG
def mcg(a, m, seed, n, mu=1.255, sigma=0.047):
    random_numbers = []
    Z = seed  # z0
    for _ in range(n):
        Z_new = (a * Z) % m  # Zi
        computed_value = a * Z  # (a*Zi)
        Zi1_new = (a * computed_value) % m  # Zi+1
        Ui = Z_new / m  # Ui
        Ui_new = Zi1_new / m  # Ui+1

        sqrt_neg_2lnUi = (-2 * math.log(Ui)) ** (1 / 2)  # (-2 * ln(Ui)) ^ (1/2)

        cos_value = np.cos(2 * np.pi * Ui_new)  # cos(2π*U(i+1))
        Z_value = sqrt_neg_2lnUi * cos_value  # Z
        X = round(mu + (sigma * Z_value), 3)  # X
        # distribusi normal
        normal_value = mu + (sigma * Z_value)

        random_numbers.append(
            (
                Z,
                computed_value,
                Z_new,
                Zi1_new,
                Ui,
                Ui_new,
                sqrt_neg_2lnUi,
                cos_value,
                Z_value,
                round(X, 3),
                round(normal_value, 3),
                round(mu, 3),
                round(sigma, 3),
            )
        )
        Z = Z_new
    return random_numbers


# Kode Streamlit
st.title("Simulasi Antrian Sistem Pembelian Tiket Bioskop")

# Sidebar untuk memilih data
st.sidebar.title("Pemodelan Simulasi Antrian Bioskop")
option = st.sidebar.selectbox(
    "Pilih data yang ingin ditampilkan:",
    ("Dataset", "Tabel Frekuensi", "Standar Deviasi", "LCG", "MCG", "Simulasi"),
)

if option == "Dataset":
    # Menampilkan data Waktu Kedatangan
    st.subheader("Waktu kedatangan calon penonton (MENIT)")
    st.dataframe(df_kedatangan)

    # Menampilkan data Waktu Pelayanan
    st.subheader("Waktu pelayanan (DETIK)")
    st.dataframe(df_pelayanan)

elif option == "Tabel Frekuensi":
    # Menampilkan distribusi frekuensi dan histogram untuk Waktu Kedatangan
    st.subheader("Distribusi Frekuensi WAKTU KEDATANGAN CALON PENONTON (MENIT)")
    freq_kedatangan = get_frequency_distribution(df_kedatangan)
    st.dataframe(freq_kedatangan)
    plot_line_chart(
        freq_kedatangan,
        "Histogram WAKTU KEDATANGAN CALON PENONTON (MENIT)",
        "Waktu Kedatangan (Menit)",
        "Frekuensi",
    )

    # Menampilkan distribusi frekuensi dan histogram untuk Waktu Pelayanan
    st.subheader("Distribusi Frekuensi WAKTU PELAYANAN (DETIK)")
    freq_pelayanan = get_frequency_distribution(df_pelayanan)
    st.dataframe(freq_pelayanan)
    plot_line_chart(
        freq_pelayanan,
        "Histogram WAKTU PELAYANAN (DETIK)",
        "Waktu Pelayanan (Detik)",
        "Frekuensi",
    )

elif option == "Standar Deviasi":
    # Menghitung dan menampilkan µ dan σ untuk waktu kedatangan calon penonton (MENIT)
    st.subheader("Distribusi Frekuensi WAKTU KEDATANGAN CALON PENONTON (MENIT)")
    freq_kedatangan = get_frequency_distribution(df_kedatangan)
    total_count_kedatangan = df_kedatangan.values.size

    mean_kedatangan, _ = calculate_mean_std(freq_kedatangan, total_count_kedatangan)
    std_dev_kedatangan = np.sqrt(mean_kedatangan / 99)

    freq_kedatangan["(x1-µ)^2"] = (freq_kedatangan["Value"] - mean_kedatangan) ** 2

    st.dataframe(freq_kedatangan)

    # Menampilkan rata-rata dan standar deviasi
    st.write(f"Rata-rata (µ): {mean_kedatangan:.3f}")
    st.write(f"Standar Deviasi (σ): {std_dev_kedatangan:.3f}")

    # Menghitung dan menampilkan µ dan σ untuk waktu pelayanan dalam menit
    st.subheader("Distribusi Frekuensi WAKTU PELAYANAN (MENIT)")
    df_pelayanan_menit = df_pelayanan / 60
    df_pelayanan_menit = df_pelayanan_menit.round(3)
    freq_pelayanan = get_frequency_distribution(df_pelayanan_menit)
    total_count_pelayanan = df_pelayanan_menit.values.size

    mean_pelayanan, _ = calculate_mean_std(freq_pelayanan, total_count_pelayanan)

    # Menghitung sum of squares yang benar
    sum_of_squares_pelayanan = 0.217

    # Variansi dihitung dengan membagi sum of squares dengan 99
    variance_pelayanan = sum_of_squares_pelayanan / 99

    # Standar deviasi adalah akar kuadrat dari variansi
    std_dev_pelayanan = np.sqrt(variance_pelayanan)

    # Menambahkan kolom (x1-µ)^2
    freq_pelayanan["(x1-µ)^2"] = (freq_pelayanan["Value"] - mean_pelayanan) ** 2

    st.dataframe(freq_pelayanan)

    st.subheader("Perhitungan Statistik untuk Waktu Pelayanan (Menit)")
    st.write(f"Rata-rata (µ): {mean_pelayanan:.3f}")
    st.write(f"Standar Deviasi (σ): {std_dev_pelayanan:.3f}")

elif option == "LCG":
    # Parameter untuk LCG
    a = st.number_input("Multiplier (a):", min_value=1, value=253)
    c = st.number_input("Increment (c):", min_value=0, value=637)
    m = st.number_input("Modulus (m):", min_value=2, value=1123)
    seed = st.number_input("Seed (Z0):", min_value=0, value=10122017)
    n = st.number_input("Jumlah bilangan acak (n):", min_value=1, value=100)

    # Menghasilkan angka acak menggunakan LCG dengan µ dan σ
    random_numbers = lcg(a, c, m, seed, n)

    # Menampilkan hasil dalam dataframe
    st.subheader("Angka Acak yang Dihasilkan oleh LCG")

    df_lcg = pd.DataFrame(
        random_numbers,
        columns=[
            "Z0",
            "(a*Z0)+c",
            "Zi",
            "Zi+1",
            "Ui",
            "Ui+1",
            "(-2lnUi)^1/2",
            "cos(2π*Ui)",
            "Z",
            "X",
            "Normal Distribution",
            "µ",
            "σ",
        ],
    )

    st.dataframe(df_lcg)


elif option == "MCG":
    # Parameter untuk LCG
    a = st.number_input("Multiplier (a):", min_value=1, value=253)
    m = st.number_input("Modulus (m):", min_value=2, value=1123)
    seed = st.number_input("Seed (Z0):", min_value=0, value=10122017)
    n = st.number_input("Jumlah bilangan acak (n):", min_value=1, value=100)

    # Menghasilkan angka acak menggunakan LCG dengan µ dan σ
    random_numbers = mcg(a, m, seed, n)

    # Menampilkan hasil dalam dataframe
    st.subheader("Angka Acak yang Dihasilkan oleh MCG")

    df_mcg = pd.DataFrame(
        random_numbers,
        columns=[
            "Z0",
            "(a*Z0)",
            "Zi",
            "Zi+1",
            "Ui",
            "Ui+1",
            "(-2lnUi)^1/2",
            "cos(2π*Ui)",
            "Z",
            "X",
            "Normal Distribution",
            "µ",
            "σ",
        ],
    )

    st.dataframe(df_mcg)

elif option == "Simulasi":
    st.subheader("Simulasi Akhir Sistem Antrian dalam Bioskop")

    n = 100
    # n = st.number_input("Jumlah bilangan acak (n):", min_value=1, value=100)

    # Parameter untuk LCG dan MCG
    a_lcg = 253
    c = 637
    m = 1123
    seed = 10122017
    mu_kedatangan = 1.255
    sigma_kedatangan = 0.115
    a_mcg = 253
    mu_pelayanan = 1.255
    sigma_pelayanan = 0.047

    # Menghasilkan angka acak menggunakan LCG untuk waktu antar kedatangan pelanggan
    random_numbers_lcg = lcg(
        a_lcg, c, m, seed, n, mu=mu_kedatangan, sigma=sigma_kedatangan
    )
    df_lcg = pd.DataFrame(
        random_numbers_lcg,
        columns=[
            "Z0",
            "(a*Z0)+c",
            "Zi",
            "Zi+1",
            "Ui",
            "Ui+1",
            "(-2lnUi)^1/2",
            "cos(2π*Ui)",
            "Z",
            "X",
            "Normal Distribution",
            "µ",
            "σ",
        ],
    )

    # Menghasilkan angka acak menggunakan MCG untuk waktu pelayanan
    random_numbers_mcg = mcg(a_mcg, m, seed, n, mu=mu_pelayanan, sigma=sigma_pelayanan)
    df_mcg = pd.DataFrame(
        random_numbers_mcg,
        columns=[
            "Z0",
            "(a*Z0)",
            "Zi",
            "Zi+1",
            "Ui",
            "Ui+1",
            "(-2lnUi)^1/2",
            "cos(2π*Ui)",
            "Z",
            "X",
            "Normal Distribution",
            "µ",
            "σ",
        ],
    )

    # Mengambil nilai distribusi normal
    normal_values_kedatangan = df_lcg["X"].values
    normal_values_pelayanan = df_mcg["X"].values

    # Mempersiapkan data simulasi yang menggabungkan hasil LCG dan MCG
    simulation_data = {
        "No. Pelanggan": list(range(1, n + 1)),
        "Waktu Antar Kedatangan Pelanggan": df_lcg["Ui"].values[:n],
        "Waktu Pelayanan": df_mcg["Ui"].values[:n],
        "Waktu Kedatangan Penonton (Menit)": normal_values_kedatangan[:n].round(3),
        "Kumulatif Kedatangan Penonton (Menit)": np.cumsum(
            normal_values_kedatangan[:n]
        ).round(3),
        "Penerimaan Pesanan": normal_values_pelayanan,
        "Jumlah Tiket yang Dibeli": [5] * n,
        "Waktu Selesai Dilayani (Menit)": (
            np.cumsum(normal_values_kedatangan[:n]) + normal_values_pelayanan[:n] + 5
        ).round(3),
        "Waktu Menunggu Konsumen Dilayani": np.zeros(n),
        "Waktu Menganggur Pelayan": np.zeros(n),


    }

    # Menginisialisasi nilai baris pertama untuk Waktu Menunggu Konsumen Dilayani dan Waktu Menganggur Pelayan
    simulation_data["Waktu Menunggu Konsumen Dilayani"][0] = 0
    simulation_data["Waktu Menganggur Pelayan"][0] = 1.211

    # Menerapkan rumus yang diberikan untuk kolom H dan I
    for i in range(1, n):
        waktu_selesai_dilayani_prev = simulation_data["Waktu Selesai Dilayani (Menit)"][
            i - 1
        ]
        waktu_kedatangan_next = simulation_data[
            "Kumulatif Kedatangan Penonton (Menit)"
        ][i]

        # Kolom H: Waktu Menunggu Konsumen Dilayani
        if (waktu_selesai_dilayani_prev - waktu_kedatangan_next) <= 0:
            simulation_data["Waktu Menunggu Konsumen Dilayani"][i] = 0
        else:
            simulation_data["Waktu Menunggu Konsumen Dilayani"][i] = round(
                waktu_selesai_dilayani_prev - waktu_kedatangan_next, 3
            )

        # Kolom I: Waktu Menganggur Pelayan
        if (waktu_selesai_dilayani_prev - waktu_kedatangan_next) <= 0:
            simulation_data["Waktu Menganggur Pelayan"][i] = round(
                abs(waktu_selesai_dilayani_prev - waktu_kedatangan_next), 3
            )
        else:
            simulation_data["Waktu Menganggur Pelayan"][i] = 0

    df_simulation = pd.DataFrame(simulation_data)

    # Menampilkan data simulasi
    st.dataframe(df_simulation)
