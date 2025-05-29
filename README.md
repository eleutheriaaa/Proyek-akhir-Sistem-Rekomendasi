# Laporan Proyek Machine Learning - Nathanael Dennis Gunawan

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
Dataset yang digunakan dalam proyek ini berasal dari Kaggle. Dataset ini berisi informasi lengkap tentang berbagai jenis smartphone, termasuk fitur teknis seperti brand, model, sistem operasi, memori, kamera, baterai, harga, serta data rating dan profil pengguna  
- **URL/Tautan Sumber Data** 
Dataset yang digunakan dalam proyek ini dapat diunduh melalui link ini: https://www.kaggle.com/datasets/meirnizri/cellphones-recommendations?select=cellphones+data.csv  
- **Jumlah Baris dan Kolom**
Dataset yang digunakan total adalah 3 dataset yang tujuan utamanya adalah untuk membuat sistem rekomendasi pemilihan cellphone, yaitu:  
	- Cellphone.csv: dataset ini terdiri dari 33 baris data dan 14 kolom yang berisi mengenai brand, model, os, dan lainnya 
 	- Rating.csv: dataset ini terdiri dari 990 baris data dan 3 kolom yang berisi rating semua pengguna
  	- User.csv: dataset ini terdiri dari 99 baris data dan 4 kolom yang berisi umur, jenis kelamin dan pekerjaan
- **Kondisi Data**
	- Missing Value: Terdapat 10 data missing value pada kolom occupation
	- Duplikat: Tidak terdapat duplikasi baris data

**Variabel-variabel dalam dataset adalah sebagai berikut:**
- cellphone_id : Kode unik yang mengidentifikasi setiap smartphone dalam dataset
- brand : Merek atau produsen smartphone 
- model : Nama atau tipe model smartphone 
- operating system : Sistem operasi yang digunakan smartphone 
- internal memory : Kapasitas penyimpanan internal smartphone dalam gigabyte (GB)
- RAM : Memori akses acak yang tersedia pada smartphone, dalam GB, mempengaruhi kecepatan pemrosesan
- performance : Ukuran atau skor performa smartphone 
- main camera : Spesifikasi kamera utama smartphone, biasanya dalam megapiksel (MP)
- selfie camera : Spesifikasi kamera depan/swatch smartphone, dalam MP
- battery size : Kapasitas baterai smartphone dalam milliampere-hour (mAh)
- screen size : Ukuran layar smartphone dalam inci
- weight : Berat smartphone dalam gram (g)
- price : Harga smartphone, biasanya dalam satuan mata uang tertentu (dalam dollar)
- release date : Tanggal atau tahun rilis smartphone
- user_id : Kode unik yang mengidentifikasi pengguna atau reviewer dalam dataset
- rating : Nilai rating yang diberikan pengguna terhadap smartphone (1-10)
- age : Usia pengguna dalam tahun
- gender : Jenis kelamin pengguna (Male/Female)
- occupation : Pekerjaan atau profesi pengguna

### **Tahapan**
- Exploratory Data Analysis (EDA)
	- Mengecek struktur data (info()), jumlah unik cellphone_id, daftar brand, sistem operasi (operating system), serta rentang harga. Sehingga dapat diketahui brand yang tersedia dan variasi harga produk
	- Melihat ringkasan statistik (describe()), total rating, dan jumlah user serta produk yang diberi rating. Sehingga dapat diketahui distribusi rating
	- Terakhir, mengecek dimensi data user untuk melihat ukuran populasi user dan informasi demografis dasar. Sehingga dapat diketahui persebaran umur, gender dan pekerjaan pengguna
- Data Pre-processing
	- Menggabungkan cellphone_id dari beberapa tabel, menghapus duplikat, dan menyatukan data rating, cellphone, dan user menjadi satu dataframe lengkap (merged_data)
	- Menggunakan .isnull().sum() untuk mengecek nilai hilang, serta groupby() untuk melihat total rating per cellphone
	- Menggabungkan data rating, nama model, dan data user menjadi all_cellphone sebagai dasar sistem rekomendasi 

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
