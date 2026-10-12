# Tugas Take-Home Sesi 05: Transfer Learning untuk Klasifikasi Citra dengan Data Terbatas

**Nama:** Vincent Christian Atmadjaya | **NIM:** 2802392646 | **Mata kuliah:** Deep Learning

Repositori ini berisi implementasi dan hasil eksperimen perbandingan *training from scratch*, *feature extraction*, dan *progressive fine-tuning* pada klasifikasi 10 ras kucing dan anjing (Oxford-IIIT Pet) menggunakan ResNet-18, beserta uji kontrol *data augmentation* dan perbandingan dengan arsitektur efisien (EfficientNet-B0 dan MobileNetV3-Small).

## Struktur Repositori

```
.
├── Tugas05_Transfer_Learning.ipynb   # notebook utama (sudah dieksekusi, output tersimpan)
├── Laporan_Tugas05.pdf               # laporan analisis (3-5 halaman)
├── README.md
└── outputs/
    ├── results.csv / results.json    # metrik mentah per eksperimen
    ├── history_all.csv               # kurva loss dan akurasi per epoch
    ├── run_config.json               # konfigurasi dan versi library
    ├── split_assignment.csv          # pembagian train/val/test per citra
    ├── table_consolidated.csv        # tabel konsolidasi B1-B5
    ├── failure_analysis.csv          # daftar kesalahan prediksi di test set
    ├── mcnemar.csv                   # uji McNemar antar-kondisi (jika sel ditambahkan)
    ├── predictions_*.npz             # prediksi test set per eksperimen
    └── figures/                      # kurva loss, failure samples, confusion matrix, Grad-CAM
```

Checkpoint model (`outputs/checkpoints/*.pt`) tidak disertakan karena ukurannya besar. Checkpoint dibuat ulang saat notebook dijalankan.

## Cara Menjalankan

1. Buka notebook di Google Colab (tombol di atas, atau unggah manual).
2. Pilih **Runtime → Change runtime type → T4 GPU**.
3. Jalankan **Runtime → Run all**. Dataset dan bobot pretrained diunduh otomatis.
4. Seluruh keluaran tersimpan di folder `outputs/`.

Untuk menjalankan secara lokal: `pip install -r requirements.txt`, lalu buka notebook dengan Jupyter. GPU sangat disarankan.

## Dataset

[Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/) (`torchvision.datasets.OxfordIIITPet`), subset 10 kelas:

- Kucing: Abyssinian, Bengal, Birman, Persian, Siamese
- Anjing: Beagle, Pug, Samoyed, Shiba Inu, Chihuahua

Split stratified 70/15/15 (train/validation/test) dengan `random_state=42`. Split dilakukan satu kali pada level citra sebelum transformasi, dan tidak ada irisan antar-split. Penetapan per citra tersimpan di `outputs/split_assignment.csv`.

## Protokol Eksperimen

| Parameter | Nilai |
|---|---|
| Arsitektur utama | ResNet-18 (`ResNet18_Weights.IMAGENET1K_V1` untuk kondisi pretrained) |
| Input | 224×224, normalisasi ImageNet |
| Batch size | 32 |
| Optimizer / loss | AdamW (weight decay 1e-2) / CrossEntropyLoss |
| Early stopping | Pada validation loss, patience 5, maksimum 60 epoch |
| Seed | 42 (Python, NumPy, PyTorch) |
| Pemilihan model | Berdasarkan validation set, bukan test set |

Catatan implementasi: pada bagian backbone yang dibekukan, layer BatchNorm dipaksa berada pada mode `eval()` agar running statistics ImageNet tidak berubah selama pelatihan.

## Kondisi Eksperimen

| Kode | Deskripsi |
|---|---|
| B1 | ResNet-18 dari inisialisasi acak (`weights=None`), tanpa augmentasi |
| B2 | Feature extraction: backbone beku, hanya `fc` baru yang dilatih, tanpa augmentasi |
| B3 | Fine-tuning `layer4` + `fc` dengan differential learning rate (1e-4 dan 1e-3), tanpa augmentasi |
| B4 | B3 dengan augmentasi (RandomResizedCrop, horizontal flip, rotasi 15°, color jitter) hanya pada train set |
| B5 | Konfigurasi terbaik (berdasarkan validation) pada EfficientNet-B0 dan MobileNetV3-Small |

## Ringkasan Hasil

Seluruh angka bersumber dari `outputs/results.csv` (satu seed; n = 300 citra untuk validation dan test).

| Kode | Trainable params | Epoch terbaik (total) | Waktu latih (s) | Val acc | Test Macro-F1 |
|---|---|---|---|---|---|
| B1 | 11.181.642 | 22 (27) | 264,7 | 0,6233 | 0,5829 |
| B2 | 5.130 | 38 (43) | 380,7 | 0,9767 | 0,9700 |
| B3 | 8.398.858 | 21 (26) | 232,4 | 0,9900 | 0,9632 |
| B4 (B3 + augmentasi) | 8.398.858 | 15 (20) | 265,1 | 0,9833 | 0,9430 |
| B5 EfficientNet-B0 | 1.142.202 | 21 (26) | 231,6 | 0,9967 | 0,9833 |
| B5 MobileNetV3-Small | 654.890 | 14 (19) | 163,1 | 0,9767 | 0,9532 |

Analisis lengkap, failure analysis, dan keterbatasan studi ada di laporan (`Laporan_Tugas05.pdf`).

## Keterbatasan

Seluruh hasil berasal dari satu seed, 10 kelas, dan 300 citra pada validation dan test, sehingga selisih kecil antar-kondisi pretrained tidak dapat dibedakan secara meyakinkan. Detail ada di laporan.

## Sitasi dan Atribusi

- Konfigurasi model, pipeline augmentasi, dan differential learning rate diadaptasi dari *starter code* pada soal tugas Sesi 05.
- Pustaka: PyTorch, torchvision, scikit-learn, pandas, matplotlib, SciPy (uji McNemar).
- Dataset: Parkhi et al., *Cats and Dogs*, CVPR 2012.

## Pernyataan Penggunaan AI

Claude (Anthropic) digunakan sebagai alat bantu untuk menyusun kerangka kode notebook, menjelaskan konsep, dan memeriksa konsistensi angka. Analisis, interpretasi, dan kesimpulan pada laporan ditulis sendiri. Rincian lengkap ada pada lampiran laporan.
