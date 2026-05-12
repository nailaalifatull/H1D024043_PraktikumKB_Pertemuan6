# Praktikum 6 — Jaringan Syaraf Tiruan (JST)

Nama : Naila Alifatul Mabruroh
NIM : H1D024043

Repositori ini berisi implementasi Python untuk dua algoritma Jaringan Syaraf Tiruan (JST):

1. **Perceptron** — untuk menyelesaikan masalah OR (bipolar)
2. **Backpropagation** — untuk menyelesaikan masalah XOR (bipolar)

---

## Penjelasan Program

### 1. Perceptron (`Perceptron.py`)

Perceptron adalah model JST sederhana dengan satu layer yang digunakan untuk proses pembelajaran terawasi (*supervised learning*). Program ini digunakan untuk menyelesaikan masalah OR bipolar.

| x1 | x2 | Target |
| -- | -- | ------ |
| 1  | 1  | 1      |
| 1  | -1 | 1      |
| -1 | 1  | 1      |
| -1 | -1 | -1     |

**Parameter:**

* Learning rate (α) = 0.1
* Epoch maksimum = 10
* Bobot dan bias awal = 0

**Output:**

* `HasilPerceptron.txt`
* `decision_boundary_epoch_N.png`

---

### 2. Backpropagation (`Backpropagation.py`)

Backpropagation adalah model JST yang memiliki beberapa layer dan digunakan untuk menyelesaikan masalah yang lebih kompleks, seperti XOR bipolar.

| x1 | x2 | Target |
| -- | -- | ------ |
| 1  | 1  | -1     |
| 1  | -1 | 1      |
| -1 | 1  | 1      |
| -1 | -1 | -1     |

**Parameter:**

* Learning rate (α) = 0.3
* Epoch maksimum = 1000
* Target error (SSE) = 0.001
* Jumlah neuron: input=2, hidden=2, output=1
* Bobot dan bias awal = random

**Output:**

* `hasilBackpropagation.txt`
* `error_plot.png`

---

## Kesalahan Kode dan Perbaikannya

Pada kode asli terdapat beberapa kesalahan yang menyebabkan program error atau hasil training menjadi kurang sesuai. Pada bagian Perceptron, terdapat masalah saat membaca nilai target sehingga proses update bobot tidak berjalan dengan benar. Selain itu, program juga sempat mengalami pembagian dengan nol ketika membuat grafik garis pemisah data pada epoch awal. Proses plotting juga membuat program berhenti sementara karena grafik harus ditutup secara manual terlebih dahulu. Setelah diperbaiki, proses training dan penyimpanan grafik dapat berjalan lebih lancar.

Pada bagian Backpropagation, terdapat masalah pada bentuk data (*shape array*) saat proses perhitungan forward dan backward propagation. Hal ini menyebabkan operasi perhitungan antar data menjadi tidak cocok dan memunculkan error. Selain itu, bentuk target output juga belum konsisten dengan hasil prediksi sehingga mempengaruhi perhitungan error. Sama seperti pada Perceptron, proses plotting grafik error juga membuat program berhenti sementara sebelum diperbaiki. Setelah dilakukan penyesuaian bentuk data dan perbaikan plotting, program dapat berjalan dengan normal dan menghasilkan output yang sesuai.

---

## Konsep Penting

**Mengapa Perceptron tidak bisa menyelesaikan XOR?**
Karena data XOR tidak dapat dipisahkan hanya dengan satu garis lurus. Perceptron hanya mampu membuat satu batas pemisah sederhana, sehingga tidak cukup untuk menangani pola XOR. Oleh karena itu digunakan Backpropagation yang memiliki hidden layer agar dapat mengenali pola yang lebih kompleks.

**Perbedaan Perceptron dan Backpropagation**

| Aspek             | Perceptron              | Backpropagation                              |
| ----------------- | ----------------------- | -------------------------------------------- |
| Struktur jaringan | Satu layer sederhana    | Memiliki beberapa layer                      |
| Kemampuan         | Untuk masalah sederhana | Untuk masalah lebih kompleks                 |
| Proses belajar    | Update bobot langsung   | Menggunakan forward dan backward propagation |
| Cocok untuk       | Data linear             | Data non-linear seperti XOR                  |
