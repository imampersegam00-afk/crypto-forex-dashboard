# RIWAYAT & CHECKPOINT TUGAS (TASK HISTORY)
> File ini berfungsi sebagai memori persisten agar AI asisten selalu mengetahui konteks, progress, dan riwayat pekerjaan tanpa harus mengulang dari awal saat token/kuota habis atau saat memulai sesi baru.

---

## 📌 PROYEK: Institutional Crypto & Forex Dashboard (Terminal Elite)
* **Direktori**: `~/crypto-forex-dashboard`
* **File Utama**: `app.py`, `templates/index.html`
* **Port**: `8080` (HTTP)
* **Lingkungan**: Termux Native (aarch64 Android), Python 3 + Flask + NumPy + Pandas (TANPA library berat/scikit-learn/tensorflow)

---

## 📋 DAFTAR CHECKLIST TUGAS & STATUS

### Tahap 1: Upgrade AI Model & Fix Bug Utama (dari `dashboard_ai_upgrade_prompt.txt`)
- [x] **1. Fix Bug HF Signal & Score**:
  - Penanganan error pada engine Hedge Fund agar tidak return `N/A` dan `0/100`.
  - Fallback terstruktur jika data pasar tidak lengkap.
- [x] **2. Model Machine Learning Statistik Asli (NumPy Murni)**:
  - Implementasi manual Logistic Regression menggunakan operasi vektor NumPy (tanpa sklearn/torch).
  - Feature engineering: RSI(14), MACD histogram, ATR ratio, Volume z-score, return momentum.
  - Sigmoid probability calculation (0.0 - 1.0) untuk klasifikasi BUY / SELL / NEUTRAL.
  - File model weights tersimpan otomatis di `ml_weights.json`.
- [x] **3. Dynamic Confidence & Accuracy**:
  - Confidence dihitung proporsional dari output probabilitas model ML (bukan hardcoded).
  - Akurasi dinamis dievaluasi dari walk-forward bar sebelumnya.
- [x] **4. Reasoning Text per Sinyal**:
  - Setiap output sinyal menyertakan alasan teknikal singkat (contoh: RSI oversold, MACD positif, konfluensi volume).
- [x] **5. Historical Prediction Log & Track Record Panel**:
  - Pencatatan otomatis ke `prediction_log.csv` saat ada sinyal baru.
  - Evaluasi win rate historis dan tabel riwayat prediksi di dashboard UI.

### Tahap 2: Perbaikan UI & Filter Konten (Request Pengguna 8-15 Oktober)
- [x] **1. Perbaikan Bug `undefined%` (Tampilan Persentase Kosong)**:
  - Perbaikan di `templates/index.html` pada elemen `sigchg`, `mob-chg-val`, `sigconf`, dan ticker marquee.
  - Menambahkan validasi `null`/`NaN` dengan fallback ke tanda strip `—` agar tidak lagi muncul teks `undefined%`.
- [x] **2. Sensor Sumber Berita & Hapus Link Eksternal**:
  - Menghapus elemen link keluar `news-src-link` (🔗 Sumber Asli) di footer kartu berita.
  - Menyensor nama sumber media menjadi `🔒 INTEL DESK [DISENSOR]`, baik di backend `app.py` maupun template.
  - Fungsi `_censor_news_text()` otomatis menyaring nama kantor berita dari headline & ringkasan.

### Tahap 3: Optimasi Performa & Cache Non-blocking
- [x] **1. Optimasi Komputasi MTF & EMA200**:
  - Menggunakan array data harga `close` yang sudah ada untuk kalkulasi indikator tanpa request HTTP berulang ke upstream Yahoo Finance.
- [x] **2. Caching Korelasi Aset Non-blocking**:
  - Eksekusi refresh data korelasi di background thread terpisah agar endpoint tidak bottleneck / lambat.
- [x] **3. Endpoint `/api/fear_greed` & Preload Resource**:
  - Menyediakan endpoint Fear & Greed index bawaan dan optimasi preload font css di browser.

### Tahap 4: Integrasi AGENTS.md & Plugin Ekosistem
- [x] **1. Konfigurasi AGENTS.md**:
  - Penambahan file panduan instruksi agen otonom di root dan direktori proyek.
- [x] **2. Instalasi & Verifikasi Plugin**:
  - Plugin `ponytail` terpasang (`agy plugin install https://github.com/DietrichGebert/ponytail`).
  - Plugin `caveman` terpasang dan aktif untuk efisiensi respon.
- [x] **3. Pull Request GitHub**:
  - Branch: `docs/add-agents-guide`
  - PR: [#1 (docs: integrasi panduan agen otonom AGENTS.md)](https://github.com/imampersegam00-afk/crypto-forex-dashboard/pull/1)

---

## 🔍 STATUS VERIFIKASI TERAKHIR
* **Syntax Check (`python -c "import ast; ..."`):** ✅ PASS
* **Import Check (`python -c "import app"`):** ✅ PASS
* **Endpoints Check (`/`, `/api/asset/...`, `/api/news`, `/api/all`, dll):** ✅ Status 200 OK
* **JavaScript Check (Node.js syntax parser):** ✅ Exit Code 0 (Bebas syntax error)

---

## 🚀 PETUNJUK AKSES & RUN SERVER
* **Menjalankan Server**:
  ```bash
  cd ~/crypto-forex-dashboard
  python app.py
  ```
* **Membuka dari Android Termux**:
  ```bash
  termux-open-url http://127.0.0.1:8080
  ```
* **Membuka dari Browser HP**: Gunakan URL lengkap `http://127.0.0.1:8080` (bukan `localhost` dan bukan `https://`).
* **Membuka dari Perangkat Lain / Tunnel**:
  ```bash
  cloudflared tunnel --url http://127.0.0.1:8080
  ```
