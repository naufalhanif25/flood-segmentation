# Segmentasi Area Banjir Menggunakan Deep Learning

## Deskripsi Proyek
Proyek ini bertujuan untuk membangun sistem **segmentasi citra** berbasis *deep learning* yang mampu mendeteksi dan menandai area banjir pada gambar. Proyek ini terinspirasi dari bencana banjir yang melanda Aceh, Sumatera Utara, dan Sumatera Barat. Model dilatih menggunakan data citra banjir beserta *mask* segmentasi untuk menghasilkan prediksi area banjir secara piksel (*pixel-wise segmentation*).

Pendekatan ini dapat digunakan untuk mendukung analisis bencana, pemantauan wilayah terdampak banjir, serta pengembangan sistem peringatan dini berbasis citra.

---

## Fitur Utama
- Segmentasi area banjir berbasis citra
- Menggunakan arsitektur **U-Net** dengan encoder *pre-trained* (ResNet)
- Mendukung *data augmentation* untuk meningkatkan generalisasi model
- Evaluasi model menggunakan berbagai metrik segmentasi
- *Early stopping* untuk mencegah *overfitting*
- Inferensi dan visualisasi hasil segmentasi pada gambar baru

---

## Teknologi yang Digunakan
- Python
- PyTorch
- segmentation-models-pytorch
- Albumentations
- OpenCV
- NumPy
- Matplotlib

---

## Persiapan Dataset
- Dataset terdiri dari pasangan gambar dan *mask* biner
- Data dibagi menjadi:
  - 80% data training
  - 20% data validasi
- File yang rusak atau memiliki ukuran tidak sesuai akan otomatis dihapus
- *Natural sorting* digunakan untuk menjaga kesesuaian antara gambar dan mask

> [!NOTE]
> Dataset yang digunakan bersumber dari platform Kaggle yang diunduh melalui tautan berikut ini: [Flood Area Segmentation](https://www.kaggle.com/datasets/faizalkarim/flood-area-segmentation)

---

## Arsitektur Model
Model menggunakan **U-Net** dengan konfigurasi:
- Encoder: ResNet34
- Pre-trained weights: ImageNet
- Input channel: 3 (RGB)
- Output channel: 1 (binary segmentation)

---

## Fungsi Loss dan Optimizer
Loss function merupakan kombinasi dari:
- Dice Loss
- Binary Cross Entropy with Logits Loss

Optimizer yang digunakan:
- Adam optimizer dengan learning rate 1e-4

---

## Proses Training
- Training dilakukan hingga maksimal 30 epoch
- Evaluasi dilakukan setiap epoch menggunakan data validasi
- Model terbaik disimpan berdasarkan nilai IoU tertinggi
- *Early stopping* digunakan jika tidak ada peningkatan performa dalam beberapa epoch

---

## Metrik Evaluasi
Model dievaluasi menggunakan metrik berikut:
- Pixel Accuracy
- Intersection over Union (IoU)
- Precision
- Recall
- F1-Score

Evaluasi dilakukan pada data validasi untuk memastikan performa model tidak mengalami overfitting.

---

## Inferensi dan Visualisasi
Model dapat digunakan untuk melakukan inferensi pada gambar baru. Output yang dihasilkan:
- Gambar asli
- Mask prediksi area banjir
- Overlay hasil segmentasi pada gambar asli

Visualisasi ditampilkan menggunakan Matplotlib dan hasil dapat disimpan ke direktori lokal.

---

## Cara Menjalankan

- Clone repository
    ```bash
    git clone https://github.com/naufalhanif25/flood-segmentation.git
    ```

- Install dependensi
    ```bash
    pip install -r requirements.txt
    ```

> [!WARNING]
> Library PyTorch (`torch`, `torchvision`) Sebaiknya diinstall sesuai dengan CUDA yang digunakan. Untuk GPU, disarankan mengikuti panduan resmi PyTorch.
> Beberapa library sebaiknya diinstall satu per satu dengan memperhatikan versi masing-masing library untuk menghindari *dependency hell*.

- Jalankan `main.py`
    ```bash
    python main.py
    ```
    atau
    ```bash
    python main.py --image "test/image.png"
    ```

> [!NOTE]
> `main.py` menjalankan inferensi segmentasi banjir pada gambar yang diinput melalui opsi `--image` dengan nilai default `test/image.png`.
> Contoh output dari program dapat dilihat pada direktori `result`.

---

## Hasil dan Evaluasi

Berikut adalah hasil evaluasi yang ditunjukkan pada epoch ke-21 sebelum proses training dihentikan oleh *Early Stopping* karena tidak mengalami peningkatan signifikan dalam beberapa epoch.

```bash
Train Loss : 0.2871
Train Acc  : 0.9334
Val Loss   : 0.3603
Val IoU    : 0.8062
Val Acc    : 0.9175
Val Prec   : 0.8877
Val Recall : 0.8989
Val F1     : 0.8921
```

Berdasarkan hasil pelatihan, model menunjukkan performa yang baik dengan nilai IoU mencapai 0.8 dan F1-score di atas 0.89 pada data validasi. Hal ini menunjukkan bahwa model mampu mengenali dan melakukan segmentasi area banjir secara konsisten dan akurat.

> [!NOTE]
> Model yang dihasilkan dari proses training dapat dilihat pada direktori `model`.

---

## Lisensi

Proyek ini didistribusikan di bawah Lisensi MIT. Lihat file `LICENSE` untuk detail selengkapnya.