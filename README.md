# Laporan Proyek Machine Learning - Nama Anda

## Project Overview

Ponsel pintar atau cellphone telah menjadi kebutuhan penting dalam kehidupan modern, digunakan untuk komunikasi, bekerja, hiburan, hingga aktivitas produktif lainnya. Banyaknya pilihan ponsel dengan berbagai merek, spesifikasi, dan fitur yang terus berkembang menjadikan proses pemilihan perangkat yang tepat menjadi semakin kompleks bagi konsumen. Informasi yang terlalu banyak (information overload) dapat menyebabkan pengguna kesulitan dalam mengambil keputusan pembelian yang sesuai dengan kebutuhan mereka

Salah satu solusi untuk membantu pengguna dalam memilih ponsel yang sesuai adalah dengan sistem rekomendasi. Sistem ini bekerja dengan menyarankan produk berdasarkan karakteristik atau preferensi pengguna sebelumnya. Salah satu pendekatan yang umum digunakan adalah content based filtering, di mana sistem menganalisis atribut produk seperti merek, sistem operasi, RAM, memori internal, dan harga untuk menemukan kemiripan antar perangkat

**Rumusan Masalah dan Penyelesaian Masalah**  
Memilih cellphone yang sesuai dengan kebutuhan merupakan tantangan yang semakin kompleks seiring berkembangnya teknologi. Maka dari itu diperlukan sistem yang mendukung sistem yang membantu sistem seleksi yang berguna untuk memengaruhi produktivitas dan kepuasan user
Penyelesaian masalah ini penting karena:  
- Efisiensi Pengambilan Keputusan: Banyaknya pilihan di pasar membuat proses pemilihan ponsel menjadi tidak efisien dan membingungkan bagi pengguna awam
- Kesesuaian Kebutuhan: Tidak semua pengguna memahami spesifikasi teknis. Tanpa bantuan sistem rekomendasi, mereka bisa memilih perangkat yang tidak sesuai dengan kebutuhan atau gaya hidup mereka
- Kepuasan Pengguna: Rekomendasi yang tidak tepat dapat menurunkan kepuasan pengguna dan menumbuhkan rasa kekecewaan setelah pembelian

Penyelesaian masalah dengan Sistem Rekomendasi berbasis Content Based Filtering dengan memanfaatkan Machine Learning:  
- Menganalisis Fitur Cellphone: Sistem memanfaatkan atribut seperti brand, sistem operasi, RAM, dan memori internal untuk menghitung kemiripan antar perangkat
- Memberikan Rekomendasi Personal: Jika seorang pengguna tertarik dengan satu merek atau model, sistem akan merekomendasikan ponsel lain yang memiliki spesifikasi serupa
- Meningkatkan Keputusan Pembelian: Dengan bantuan rekomendasi, pengguna bisa lebih percaya diri dalam memilih ponsel yang sesuai dengan kebutuhannya

[1] Menurut Anggoro dan Izzatillah (2022), sistem rekomendasi merupakan salah satu cabang dari machine learning yang lebih spesifik karena tidak hanya memprediksi nilai berdasarkan input baru, tetapi lebih berfokus pada penyediaan daftar produk yang sesuai dengan preferensi pengguna berdasarkan riwayat data sebelumnya  
[2] Menurut Nurhidayat dan Zulfikar (2020), penerapan content-based filtering dalam sistem rekomendasi smartphone terbukti efektif terutama ketika data interaksi pengguna masih terbatas. Mereka menyatakan bahwa pendekatan ini tetap dapat memberikan hasil rekomendasi yang relevan meskipun belum tersedia banyak ulasan atau riwayat pembelian. Dalam studi perbandingan yang dilakukan, pendekatan content-based filtering menghasilkan akurasi yang lebih tinggi dibandingkan dengan demographic filtering, karena mampu mempertimbangkan kesamaan fitur perangkat yang lebih spesifik dan relevan  

[1] Anggoro, M. V., & Izzatillah, M. (2022). Sistem Rekomendasi Musik dengan Metode Collaborative Filtering Berbasis Android. STRING (Satuan Tulisan Riset dan Inovasi Teknologi), 7(1), 1-8.  
[2] Nurhidayat, A., & Zulfikar, M. (2020). Perbandingan metode demographic dan content-based filtering pada sistem rekomendasi smartphone Android. Jurnal E-Proc, 7(3), 456–462.  

## Business Understanding

### Problem Statements

- Apa yang menjadi kesulitan utama pengguna dalam memilih smartphone?  
Pengguna sering mengalami kebingungan karena terlalu banyaknya pilihan smartphone dengan berbagai fitur dan harga yang berbeda  
- Mengapa sistem rekomendasi yang ada belum optimal dalam memberikan rekomendasi yang sesuai?  
Karena sebagian besar sistem belum memanfaatkan fitur spesifik perangkat secara maksimal sehingga rekomendasi yang diberikan kurang relevan dengan kebutuhan pengguna  
- Bagaimana keterbatasan data interaksi pengguna mempengaruhi kualitas rekomendasi?  
Minimnya data riwayat interaksi pengguna seperti rating dan pembelian membuat metode collaborative filtering kurang efektif  

### Goals

- Membantu pengguna dalam proses pemilihan smartphone yang tepat dan sesuai dengan preferensi mereka  
Proyek ini bertujuan mengembangkan sistem rekomendasi yang mampu menyajikan daftar smartphone yang paling relevan berdasarkan fitur dan kebutuhan pengguna, sehingga dapat mengurangi kebingungan dan mempercepat pengambilan keputusan
- Meningkatkan akurasi dan relevansi rekomendasi dengan memanfaatkan metode content-based filtering  
Dengan fokus pada kesamaan fitur smartphone, sistem akan menghasilkan rekomendasi yang lebih personal dan sesuai dengan karakteristik yang diinginkan pengguna, sehingga meningkatkan kepuasan pengguna terhadap hasil rekomendasi  
- Mengatasi masalah keterbatasan data interaksi pengguna dengan pendekatan content-based filtering  
Proyek ini mengimplementasikan teknik yang efektif meskipun data riwayat pengguna minim, sehingga sistem dapat tetap memberikan rekomendasi yang bermakna tanpa bergantung pada banyak data interaksi pengguna sebelumnya

    ### Solution statements
- Menggunakan teknik TF-IDF (Term Frequency-Inverse Document Frequency) untuk mengubah fitur teks menjadi representasi numerik  
	- Menggunakan TfidfVectorizer dari scikit-learn untuk mengolah data teks kolom model smartphone  
	- Menghitung bobot TF-IDF untuk setiap kata yang muncul dalam nama model, sehingga fitur teks dapat direpresentasikan dalam bentuk matriks numerik  
Tujuan: Mengubah data teks menjadi representasi numerik yang dapat dianalisis secara matematis sehingga memudahkan pengukuran kemiripan antar smartphone berdasarkan deskripsi model  
- Mengukur sudut kosinus antara dua vektor fitur setelah melakukan TF-IDF  
	- Menghitung cosine similarity antar matriks TF-IDF yang mewakili masing-masing smartphone  
	- Sistem merekomendasikan smartphone yang memiliki nilai similarity tertinggi terhadap smartphone input, artinya smartphone tersebut paling mirip secara fitur model  
Tujuan: Memberikan rekomendasi smartphone yang paling mirip berdasarkan fitur teks model, membantu pengguna menemukan alternatif smartphone yang relevan dengan preferensi mereka  

## Data Understanding
Paragraf awal bagian ini menjelaskan informasi mengenai jumlah data, kondisi data, dan informasi mengenai data yang digunakan. Sertakan juga sumber atau tautan untuk mengunduh dataset. Contoh: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Restaurant+%26+consumer+data).

Selanjutnya, uraikanlah seluruh variabel atau fitur pada data. Sebagai contoh:  

Variabel-variabel pada Restaurant UCI dataset adalah sebagai berikut:
- accepts : merupakan jenis pembayaran yang diterima pada restoran tertentu.
- cuisine : merupakan jenis masakan yang disajikan pada restoran.
- dst

**Rubrik/Kriteria Tambahan (Opsional)**:
- Melakukan beberapa tahapan yang diperlukan untuk memahami data, contohnya teknik visualisasi data beserta insight atau exploratory data analysis.

## Data Preparation
Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

## Modeling
Tahapan ini membahas mengenai model sisten rekomendasi yang Anda buat untuk menyelesaikan permasalahan. Sajikan top-N recommendation sebagai output.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menyajikan dua solusi rekomendasi dengan algoritma yang berbeda.
- Menjelaskan kelebihan dan kekurangan dari solusi/pendekatan yang dipilih.

## Evaluation
Pada bagian ini Anda perlu menyebutkan metrik evaluasi yang digunakan. Kemudian, jelaskan hasil proyek berdasarkan metrik evaluasi tersebut.

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.
