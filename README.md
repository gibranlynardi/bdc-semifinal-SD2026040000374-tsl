# Audit stratifikasi tersembunyi (hidden stratification): paket kode pengelompokan unsupervised

Paket ini mengulang dari awal pengelompokan foto e-waste kelas Elektronik yang dipakai di makalah. Semua langkah dijalankan oleh satu perintah, dan hasilnya dapat dicek terhadap artefak analisis asli.

Alurnya:

1. Muat embedding DINOv3 yang dibekukan, lalu pilih 3.607 foto Elektronik yang tersisa setelah deduplikasi.
2. Jalankan beberapa konfigurasi kandidat (Leiden, K-Means, UMAP+HDBSCAN). Untuk tiap kandidat hitung validitas internal dan stabilitasnya.
3. Pilih satu konfigurasi dengan aturan tanpa label yang ditulis sebelum evaluasi.
4. Audit klaster terpilih terhadap nama file: nama yang terpecah ke beberapa klaster, klaster tanpa nama, dan stabilitas keduanya.
5. Ekspor tabel per foto untuk pembuatan gambar.

## Label manusia tidak dipakai

* Label manusia (anotasi tim dan kunci anotasi) hanya untuk evaluasi. Paket ini tidak membaca file label manusia sama sekali. Kolom yang dibaca dari file indeks hanya `file`, `cls`, `provenance`, `device_label` dan `keep`.
* Pengelompokan dan pemilihan konfigurasi (langkah 1 sampai 3) tidak memakai label apa pun.
* Nama dari nama file (`device_label`, 10 jenis perangkat) hanya ada untuk 2.332 foto katalog. Nama ini baru dipakai di tahap audit (langkah 4), setelah pilihan dibuat, untuk menjelaskan isi klaster.

## Isi folder

| Path | Isi |
|---|---|
| `config.yaml` | Semua parameter: path, seed, k=15, resolusi 1,0, daftar kandidat, aturan pilihan, ambang audit |
| `run_all.py` | Menjalankan seluruh pipeline dan menulis `outputs/` |
| `verify.py` | Membandingkan `outputs/` dengan artefak analisis asli di `reference/` |
| `notebook.ipynb` | Langkah yang sama dengan penjelasan singkat, sudah dieksekusi |
| `src/data.py` | Memuat embedding dan indeks, memilih baris, membuat fitur (`raw` dan `defl20`) |
| `src/cluster.py` | Graf kNN scanpy + Leiden, K-Means, UMAP+HDBSCAN |
| `src/validity.py` | Silhouette kosinus, Davies-Bouldin, Calinski-Harabasz, stabilitas resampling dan seed |
| `src/select.py` | Aturan pilihan tanpa label |
| `src/audit.py` | Voting nama kNN, tabel silang, skor pecah, klaster tanpa nama, stabilitas 25 pengulangan |
| `src/figures.py` | Tata letak ForceAtlas2 dan ekspor `clusters.csv` (gambar dibuat skrip lain) |
| `src/pipeline.py` | Empat tahap yang dipakai bersama oleh `run_all.py` dan notebook |
| `src/utils.py` | Konfigurasi, path, log, versi pustaka, SHA-256 |
| `tests/test_core.py` | Uji kecil pada data mainan (`python tests/test_core.py`) |
| `reference/` | Artefak analisis asli, hanya dibaca oleh `verify.py` (dan oleh opsi `layout.source: reference`) |
| `requirements.txt` | Versi pustaka yang dipakai saat verifikasi |
| `extraction/kaggle_embeddings_full.py` | Skrip asli yang menghasilkan embedding input (lihat bagian Input). Untuk dokumentasi/provenance saja, tidak dijalankan atau diverifikasi oleh paket ini |

## Input

Satu-satunya input adalah dua file hasil ekstraksi fitur. Path ditulis relatif terhadap folder proyek `SD/`.

| File | Isi |
|---|---|
| `SD/work/results_embedding_2/emb_full_A144.npy` | 26.527 x 768, float16. Embedding DINOv3 ViT-B/16 (`facebook/dinov3-vitb16-pretrain-lvd1689m`) pada 144 px, tanpa fine-tuning, satu kali jalan untuk seluruh korpus |
| `SD/work/results_embedding_2/emb_full_index.csv` | Satu baris per embedding, urutan sama. Kolom yang dipakai: `file`, `cls`, `provenance`, `device_label`, `keep` |

Ekstraksi embedding butuh GPU dan akses model di Hugging Face, jadi tidak diulang di paket ini. `run_log.txt` mencatat SHA-256 kedua file input supaya versi input yang dipakai dapat dipastikan. Skrip asli yang menghasilkan kedua file ini disertakan apa adanya di `extraction/kaggle_embeddings_full.py` untuk keperluan audit (model, resolusi, pooling, preprocessing), bukan untuk dijalankan ulang di sini.

SHA-256 input saat verifikasi:

* `emb_full_A144.npy`: `9efcdce5c09091b70dc14518c501b72f42d19c00e3ad25a7e45d3b84c236dee8`
* `emb_full_index.csv`: `65fa1dacfc662ff59416f13ea79a63be13161aaadca99dbabcf88b13fddb3f9c`

## Cara menjalankan

### Lokal

Diuji dengan Python 3.11.15.

```bash
pip install -r requirements.txt
python run_all.py --root /path/ke/SD                  # profil full
python run_all.py --root /path/ke/SD --profile quick  # profil quick
python verify.py                                      # cek terhadap analisis asli
```

Folder `SD/` dicari dengan urutan berikut: opsi `--root`, variabel lingkungan `SD_ROOT`, lalu `paths.root` di `config.yaml`. Nilai bawaannya `..`, artinya paket diletakkan di `SD/pipeline/`.

### Kaggle

1. Unggah folder `results_embedding_2` sebagai Kaggle Dataset dengan struktur `SD/work/results_embedding_2/`.
2. Unggah folder paket ini sebagai dataset kedua, atau salin ke `/kaggle/working/pipeline`.
3. Di notebook Kaggle (CPU sudah cukup, GPU tidak perlu):

```bash
cp -r /kaggle/input/<dataset-paket>/pipeline /kaggle/working/
cd /kaggle/working/pipeline
pip install -r requirements.txt
python run_all.py --root /kaggle/input/<dataset-embedding>/SD
python verify.py
```

Pin versi di `requirements.txt` penting. Versi scanpy atau igraph yang lain dapat memberi nomor klaster atau batas klaster yang sedikit berbeda, karena generator acak dan nilai bawaan bisa berubah antarversi.

## Waktu jalan

Diukur di mesin verifikasi dengan 1 thread per proses (`run_all.py` mengunci `OMP_NUM_THREADS=1`).

| Profil | Kandidat yang dijalankan ulang | Waktu |
|---|---|---|
| `quick` | 6 (Leiden r=0,5 dan r=1,0 pada `raw` dan `defl20`, K-Means k=10 dan k=20 pada `raw`) | sekitar 3,5 menit |
| `full` (bawaan) | 10 (ditambah UMAP+HDBSCAN pada `raw` dan `defl20`, K-Means k=10 dan k=20 pada `defl20`) | sekitar 22 menit |
| notebook | profil `quick` | sekitar 3,5 menit |

Rincian per kandidat ada di `outputs/run_log.txt` dan `outputs/run_log.json`. Tata letak ForceAtlas2 memakan sekitar 90 detik. Klaster terpilih sama di kedua profil.

TEMI (Adaloglou dkk., BMVC 2023) tidak dilatih ulang. Angka validitasnya disalin dari run asli ke `config.yaml` dan ditandai `recorded` di `validity.csv`. Menurut aturan pilihan, TEMI tidak mungkin terpilih karena stabilitas resamplingnya 0,53 atau tidak ada.

## Output

Semua file ditulis ke `outputs/`.

| File | Isi |
|---|---|
| `clusters.csv` | Satu baris per foto (3.607): `file`, `thumb` (nama thumbnail .jpg), `provenance` (catalogue atau scraped), `cluster`, `filename_name` (hanya katalog), `pred_name` dan `pred_conf` (voting 15 tetangga katalog, hanya scraped), `confident_name` (nama file untuk katalog, nama voting bila `pred_conf` >= 0,6 untuk scraped), `fa2_x`, `fa2_y` (posisi ForceAtlas2 graf kNN mentah) |
| `validity.csv` | Satu baris per kandidat: jumlah klaster, porsi noise, porsi klaster terbesar, silhouette kosinus, Davies-Bouldin, Calinski-Harabasz, stabilitas resampling (rata-rata dan SD ARI, 20 subsampel 80%), stabilitas seed (6 seed), kelayakan, peringkat dan pilihan |
| `choice.md` | Aturan pilihan, konfigurasi terpilih dan tabel kandidat yang layak |
| `crosstab_catalogue.csv` | Tabel silang klaster x nama file, foto katalog saja |
| `split_scores.csv` | Per nama, dalam dua pandangan (nama file katalog, dan nama yakin katalog + scraped): skor pecah (entropi dibagi ln 19), jumlah klaster efektif, klaster yang memuat minimal 10% nama itu, penanda `split`, dan klaster yang nama mayoritasnya nama itu |
| `orphans.csv` | Per klaster: ukuran, porsi katalog, nama mayoritas yakin dan porsinya, nama kedua, penanda `orphan` (nama mayoritas yakin menutup kurang dari 50% klaster) |
| `stability.csv` | Per klaster pada 25 pengulangan (20 subsampel + 5 seed): rata-rata dan minimum porsi anggota di klaster pengulangan teratasnya, dan porsi pengulangan dengan nilai minimal 0,60. Kolom `group` memberi nama kelompok yang dibahas di makalah |
| `stability_pairs.csv` | Per pasangan pecahan (CRT vs layar datar, baterai kecil vs aki, smartphone vs ponsel fitur): porsi pengulangan yang tetap memisahkan keduanya |
| `run_log.txt`, `run_log.json` | Versi pustaka, SHA-256 input dan output, waktu per tahap dan per kandidat |
| `verification.txt`, `verification_contingency.csv` | Hasil `verify.py` |
| `repro_check.txt` | Catatan uji dua salinan baru: waktu jalan, SHA-256 dan perbandingan byte |
| `cache/` | Label semua pengulangan per kandidat (npz). Dipakai ulang bila parameter dan input sama. Hapus atau pakai `--no-cache` untuk menghitung ulang |

## Hasil verifikasi

Dicek pada 25 September 2026 dengan versi di `requirements.txt`. Jalankan `python verify.py` untuk mengulang cek 1 sampai 5.

1. Label Leiden r=1,0 dari pipeline identik dengan label analisis asli (`reference/raw__Leiden_r1.0.npy`). ARI 1,000, nomor klaster sama, dan tabel kontingensi 19 x 19 hanya punya 19 sel tidak nol di diagonal (`outputs/verification_contingency.csv`).
2. Aturan tanpa label memilih `raw__Leiden_r1.0`, sama dengan pilihan asli (`reference/choice_r3.md`). Pilihan ini tipis. K-Means k=10 hanya 0,003 di bawahnya dalam stabilitas resampling (0,899 lawan 0,902), jadi keputusan jatuh ke aturan silhouette (0,200 lawan 0,213).
3. Sepuluh kandidat yang dijalankan ulang memberi 260 pelabelan (per kandidat 1 utama, 20 subsampel dan 5 seed) yang semuanya identik dengan run asli. Angka di `validity.csv` sama dengan `internal.csv` asli sampai 6 desimal.
4. Voting nama untuk 1.275 foto scraped identik dengan run asli, dan posisi ForceAtlas2 identik dengan `reference/pos_raw.npy`.
5. Stabilitas 25 pengulangan sama dengan keluaran `split_stability.py` asli.
6. Dua salinan baru folder ini (tanpa `outputs/` dan tanpa cache) dijalankan terpisah dengan profil `full`. Kedelapan tabel output identik byte demi byte, termasuk `clusters.csv` (SHA-256 berawalan `1155f59e`). Profil `quick` memberi `clusters.csv` yang sama. Ringkasannya ada di `outputs/repro_check.txt`.

## Catatan hasil audit

Angka berikut keluar langsung dari aturan di `config.yaml`, tanpa penyesuaian.

* Nama yang terpecah ke minimal dua klaster (masing-masing minimal 10% dari nama itu, foto katalog): Television (klaster 12 50,0% dan 6 48,4%), Washing_Machine (11 64,3% dan 15 34,0%), battery (8 82,7% dan 18 16,8%).
* Pecahan Mobile tidak lolos ambang 10%. Klaster 16 (ponsel fitur) hanya 21 foto, dan foto Mobile di dalamnya hanya 4,5% dari semua foto yang bernama Mobile secara yakin. Pecahan ini terlihat dari sisi klaster (klaster 16 berisi 95% Mobile dan terpisah dari klaster 3 di semua 25 pengulangan), bukan dari skor pecah.
* Aturan klaster tanpa nama menandai klaster 0 (tumpukan) dan 17 (stopkontak dan kabel). Klaster 2 dan 4 (laptop menurut pemeriksaan medoid) tidak ditandai. Voting kNN hanya bisa memilih salah satu dari 10 nama file, sehingga keduanya diberi nama Keyboard (73% klaster) dan Mobile (60% klaster). Status tanpa nama untuk laptop bersandar pada pemeriksaan visual, bukan pada aturan ini.
* Pada 25 pengulangan, kelompok baterai kecil (klaster 8) bertahan di 24 dari 25 (96%). Kelompok lain yang dibahas di makalah bertahan di 25 dari 25. Klaster paling tidak stabil adalah klaster 5 (Player), yang bertahan di 16 dari 25.

## Determinisme

* Graf kNN scanpy dihitung eksak (bukan aproksimasi) karena jumlah foto di bawah 4.096. Di atas batas itu scanpy beralih ke pynndescent yang aproksimatif. Leiden (igraph), K-Means, UMAP dan PCA memakai seed tetap dari `config.yaml`.
* Subsampel diambil sekali dengan `numpy.random.default_rng(12345)` dan dipakai bersama oleh semua kandidat.
* Semua CSV ditulis dengan format angka tetap (6 desimal) dan tanpa cap waktu, sehingga dua run memberi file yang identik byte demi byte. Cap waktu hanya ada di `run_log`.
* Label K-Means dan Leiden sama pada 1 dan 2 thread (dicek untuk K-Means k=20 dan Leiden r=1,0).
* Yang tidak diulang: pelatihan TEMI (angka tercatat saja) dan ekstraksi embedding (input tetap, dicek lewat SHA-256).

## Pemeriksaan ketahanan: estimasi subkelas GEORGE (Langkah 1)

Setelah `run_all.py` selesai, jalankan:

```
python george_check.py --root /path/ke/SD
```

Skrip ini mengikuti Langkah 1 GEORGE (Sohoni dkk., NeurIPS 2020): foto dipisah per nama kasar, lalu tiap kelompok diklaster sendiri. Jumlah klaster k dari 2 sampai 10 dipilih dengan silhouette rata-rata per klaster, lalu dilakukan overclustering dengan faktor 5. Ada dua varian. Varian `bit` memakai fitur mentah dan k-means, setara George-BiT untuk fitur pralatih beku. Varian `umap` memakai UMAP 2 dimensi dan GMM, yaitu bawaan makalah. Langkah 2 GEORGE (pelatihan GDRO) tidak dipakai karena tidak ada model yang dilatih.

Hasilnya tersimpan di `outputs/george_step1.csv` dan `outputs/v_measure_catalogue.csv`. Pemeriksaan ini hanya untuk ketahanan dan tidak ikut memilih konfigurasi utama.
