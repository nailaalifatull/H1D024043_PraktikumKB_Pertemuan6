# Import library
import numpy as np
import matplotlib.pyplot as plt

# Buat kelas Backpropagation
class Backpropagation:
    # Simpan learning rate, epoch, dan target error dalam konstruktor
    # serta inisialisasi bobot dan bias awal random
    def __init__(self, alpha, epoch, target_error):
        self.alpha = alpha
        self.epoch = epoch
        self.target_error = target_error
        self.n_input = 2
        self.n_hidden = 2
        self.n_output = 1
        self.w_hidden = np.random.rand(self.n_input, self.n_hidden)
        self.b_hidden = np.random.rand(1, self.n_hidden)
        self.w_output = np.random.rand(self.n_hidden, self.n_output)
        self.b_output = np.random.rand(1, self.n_output)

    # Fungsi menerapkan fungsi aktivasi sigmoid bipolar (tanh)
    def bi_sigmoid(self, x):
        return np.tanh(x)

    # Fungsi turunan sigmoid bipolar (asumsi x = output tanh)
    def deriv_bi_sigmoid(self, x):
        return 1 - x ** 2

    # Fungsi membuat simulasi penurunan SSE setiap epoch
    def plot_error(self, x, epoch):
        plt.figure()
        plt.plot(range(1, epoch + 1), x, linestyle='-', color='b', label='Error')
        final_error = x[-1]
        # FIX: xytext disesuaikan agar tidak error saat epoch kecil
        offset = max(1, len(x) * 0.2)
        plt.annotate(
            f'Epoch {epoch}, Error: {final_error:.4f}',
            xy=(epoch, final_error),
            xytext=(max(1, epoch - offset), final_error + 0.05),
            arrowprops=dict(facecolor='black', arrowstyle="->"),
            fontsize=10,
            color='red'
        )
        plt.title('Perbaikan Error Setiap Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Sum Square Error(SSE)')
        plt.grid(True)
        plt.legend()
        plt.savefig("error_plot.png")
        plt.close()

    # Fungsi utama Backpropagation
    def fit(self, X, t):
        errors_per_epoch = []

        # Menyimpan hasil pada hasilBackpropagation.txt
        with open("hasilBackpropagation.txt", "w") as f:
            f.write("Masalah XOR dengan Backpropagation\n")
            f.write("-----------------------------------\n")
            f.write(f"Input :\n{X}\n")
            f.write(f"Target :\n{t}\n\n")
            f.write(f"Bobot awal hidden layer :\n{self.w_hidden}\n\n")
            f.write(f"Bias awal hidden layer :\n{self.b_hidden}\n\n")
            f.write(f"Bobot awal output layer :\n{self.w_output}\n\n")
            f.write(f"Bias awal output layer :\n{self.b_output}\n\n")
            f.write(f"Learning rate : {self.alpha}\n")
            f.write(f"Max Epoch     : {self.epoch}\n\n")

            # Iterasi Backpropagation (epoch)
            for epoch in range(self.epoch):
                f.write("---------------------------------------------\n")
                f.write(f"Epoch {epoch + 1}/{self.epoch}\n")
                f.write("------------\n")
                total_error = 0
                count = 1
                output = np.array([])

                # Iterasi setiap pasang matriks input dengan targetnya
                for xi, target in zip(X, t):
                    f.write(f"Data ke-{count}\n")
                    f.write("----------\n")
                    f.write("------------ Forward Propagation ------------\n")

                    # FIX: xi perlu direshape menjadi (1, n_input) agar dot product benar
                    xi_reshaped = xi.reshape(1, -1)

                    # Operasikan input dengan hidden layer
                    h_in = np.dot(xi_reshaped, self.w_hidden) + self.b_hidden
                    f.write(f"Operasi input ke hidden layer:\n{h_in}\n\n")

                    # Aktivasi hidden layer dengan fungsi tanh
                    h = self.bi_sigmoid(h_in)
                    f.write(f"Aktivasi hidden layer:\n{h}\n\n")

                    # Operasikan hidden layer dengan output layer
                    y_in = np.dot(h, self.w_output) + self.b_output
                    f.write(f"Operasi hidden ke output layer:\n{y_in}\n\n")

                    # Aktivasi output layer dengan fungsi tanh
                    y = self.bi_sigmoid(y_in)
                    output = np.append(output, y)
                    f.write(f"Aktivasi output layer:\n{y}\n")
                    f.write("------------ Backward Propagation ------------\n")

                    # Hitung error output layer terhadap target
                    # FIX: target direshape agar dimensinya cocok dengan y
                    target_reshaped = target.reshape(1, -1)
                    error = target_reshaped - y
                    total_error += np.sum(error ** 2)
                    f.write(f"Error:\n{error}\n\n")

                    # Delta output layer
                    d_y = error * self.deriv_bi_sigmoid(y)
                    f.write(f"Delta output (d_y):\n{d_y}\n\n")

                    # Hitung error hidden layer
                    error_h = np.dot(d_y, self.w_output.T)
                    f.write(f"Error hidden layer (error_h):\n{error_h}\n\n")

                    # Delta hidden layer
                    d_h = error_h * self.deriv_bi_sigmoid(h)
                    f.write(f"Delta hidden layer (d_h):\n{d_h}\n\n")

                    # Perbaiki bobot dan bias output layer
                    self.w_output += np.dot(h.T, d_y) * self.alpha
                    f.write(f"Bobot output layer baru (w_output):\n{self.w_output}\n\n")

                    self.b_output += np.sum(d_y, axis=0, keepdims=True) * self.alpha
                    f.write(f"Bias output layer baru (b_output):\n{self.b_output}\n\n")

                    # FIX: xi direshape menjadi (n_input, 1) untuk dot product yang benar
                    self.w_hidden += np.dot(xi_reshaped.T, d_h) * self.alpha
                    f.write(f"Bobot hidden layer baru (w_hidden):\n{self.w_hidden}\n\n")

                    self.b_hidden += np.sum(d_h, axis=0, keepdims=True) * self.alpha
                    f.write(f"Bias hidden layer baru (b_hidden):\n{self.b_hidden}\n")
                    f.write("---------------------------------------------\n")
                    count += 1

                # Hitung rata-rata SSE per epoch
                average_error = total_error / len(X)
                errors_per_epoch.append(average_error)
                f.write(f"Output : {output.reshape(1, 4)}\n")
                f.write(f"Sum Square Error(SSE) epoch ke-{epoch + 1}: {average_error}\n")

                # Cek kondisi berhenti
                if average_error < self.target_error or epoch + 1 == self.epoch:
                    f.write("-" * 66 + "\n")
                    f.write(f"Pelatihan berhenti pada epoch ke-{epoch + 1} karena ")
                    if epoch + 1 != self.epoch:
                        f.write("Sum Square Error(SSE) mencapai target.\n")
                    else:
                        f.write("max epoch tercapai.\n")
                    f.write(f"Bobot akhir hidden layer :\n{self.w_hidden}\n\n")
                    f.write(f"Bias akhir hidden layer  :\n{self.b_hidden}\n\n")
                    f.write(f"Bobot akhir output layer :\n{self.w_output}\n\n")
                    f.write(f"Bias akhir output layer  :\n{self.b_output}")

                    self.plot_error(errors_per_epoch, epoch + 1)
                    break

        print(f"Selesai! Hasil disimpan di hasilBackpropagation.txt")
        print(f"Bobot akhir hidden : {self.w_hidden}")
        print(f"Bobot akhir output : {self.w_output}")